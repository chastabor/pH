"""`ctx.jobs` — background work with a handle, a cancel and a completion.

Anything long-running that is not a turn: a `/refine` planner pass, a watcher, a
background build. A job is deliberately *not* a tool call — it outlives the step
that started it, so it needs its own identity and its own cancellation rather
than borrowing the turn's.

**A job is an effect of the scope that owns it (I2).** It outlives the *step*,
not the agent or the session: a subagent's drive job belongs to the delegation, a
`/refine` pass to the session that asked, a daemon sweeper to the process. So
`start` takes a `scope=` like every other registration here, and disposing that
scope cancels the job and drops its entry. Without an owner the table only ever
grew, and the bound would have had to be a number somebody picked.

Two halves of "this job is over", deliberately distinct:

* **abandoned** — the owning scope went away while the work was still running.
  Cancel it, then forget it.
* **released** (`forget`) — the owner knows the work is finished and wants the
  entry gone. Forget it, cancel nothing. A job whose own body triggers its
  owner's teardown would otherwise report `cancelled` for work that completed.

Cancellation is cooperative: `Job.cancel()` sets a token, and a body that never
reads it will run to completion regardless — `ctx.drain()` still waits for it.

@module ph.seams.jobs
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from ..cancel import CancelToken
from ..cordis import Context, Disposer, events, maybe_await, plugin

__all__ = ["Job", "JobService", "JobState", "apply"]

log = logging.getLogger("ph.seams.jobs")

JobState: TypeAlias = Literal["running", "done", "failed", "cancelled"]

events.declare("job/started", "emit", owner="ph.seams.jobs", doc="A background job began.")
events.declare("job/settled", "emit", owner="ph.seams.jobs", doc="A background job finished.")


@dataclass(slots=True)
class Job:
    """One background job."""

    id: str
    kind: str
    label: str
    token: CancelToken
    state: JobState = "running"
    result: Any = None
    error: BaseException | None = None
    release: Disposer | None = None
    """Deregisters this job's effect from its owning scope. Set by `start`, and
    called by `JobService.forget` — not by a caller directly, because dropping the
    entry and deregistering the effect have to happen together."""

    def cancel(self) -> None:
        self.token.cancel("job cancelled")


@dataclass(slots=True)
class JobService:
    """The service published as `ctx.jobs`."""

    ctx: Context
    _jobs: dict[str, Job] = field(default_factory=dict)
    _scope: Any = None

    def bind(self, task_group: Any) -> None:
        """Adopt the task group jobs run in.

        Optional. Without one, a job runs on `ctx.detach` — the pool
        `ctx.drain()` awaits — which is still honest for a headless one-shot
        (nothing is dropped; shutdown waits) and, unlike running the body
        inline, does not make `start()` block until the job finishes. A job that
        outlives the step that started it is the entire point of the seam, so
        `start` returning early is the contract rather than an optimization.
        """
        self._scope = task_group

    async def start(
        self,
        *,
        kind: str,
        label: str,
        run: Callable[[Job], Any],
        scope: Context | None = None,
    ) -> Job:
        """Start one background job, owned by `scope` (default: this seam's).

        `scope` is what bounds the job's lifetime: the delegation, the session,
        the process. Disposing it abandons the job — cancelled if still running,
        and dropped from the table either way.
        """
        job = Job(id=f"{kind}-{secrets.token_hex(4)}", kind=kind, label=label, token=CancelToken())
        self._jobs[job.id] = job

        def enter() -> Disposer:
            def abandon() -> None:
                # The entry's presence *is* the "still owned" flag: `forget` pops
                # it before deregistering, so an owner that released a finished
                # job cannot have it reported as cancelled here.
                if self._jobs.pop(job.id, None) is None:
                    return
                if job.state == "running":
                    job.cancel()

            return abandon

        job.release = await self.ctx.owner_for(scope).effect(enter, label=f"job({job.id})")
        self.ctx.emit("job/started", job, contained=True)

        async def body() -> None:
            try:
                job.result = await maybe_await(run(job))
                job.state = "cancelled" if job.token.cancelled else "done"
            except Exception as error:
                job.state = "failed"
                job.error = error
                log.debug("ph.seams.jobs: job %s failed", job.id, exc_info=True)
            finally:
                self.ctx.emit("job/settled", job, contained=True)

        if self._scope is not None:
            self._scope.start_soon(body)
        else:
            self.ctx.detach(body(), label=f"job {job.id}")
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return list(self._jobs.values())

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.cancel()
        return True

    def forget(self, job_id: str) -> bool:
        """Drop a finished job's entry without cancelling it.

        For an owner that knows the work is done — a delegation whose child has
        settled — so the table does not hold one entry per unit of past work for
        the life of the scope. Deregisters the effect too, so the scope stops
        holding a teardown for something already over.
        """
        job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        if job.release is not None:
            job.release()
        return True


@plugin("jobs-local")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the local job runner."""
    ctx.provide("jobs", JobService(ctx=ctx))
