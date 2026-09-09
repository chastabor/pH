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

**Slots are how a producer bounds its own concurrency** (`slot=`). Work that
arrives faster than it should run is queued rather than refused: the caller asked
for all of it, and a refusal answers a question about resources with one about
intent. The primitive is here rather than in each producer because "wait for a
free slot" is forty lines of subtle async — a cancel scope around the wait, a
release on every ending, a table that has to be dropped — and every one of those
subtleties is discovered once and then re-discovered by the next producer. A
subagent provider, a background build and a watcher all queue the same way, and a
second provider inherits it instead of re-deriving it.

@module ph.seams.jobs

"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import anyio
from pydantic import Field

from ..cancel import CancelToken
from ..cordis import Context, Disposer, events, maybe_await, plugin
from ..keys import JOBS
from ..wire import WireModel

__all__ = [
    "CHILDREN_KIND",
    "UNSETTLED",
    "Config",
    "Job",
    "JobService",
    "JobState",
    "Slot",
    "apply",
]

log = logging.getLogger("ph.seams.jobs")

JobState: TypeAlias = Literal["queued", "running", "done", "failed", "cancelled"]
"""`queued` is admitted but waiting for a slot — see `JobService.start`'s `slot=`.

A state rather than an absence, because "not started yet" and "not started at
all" are different things to whoever is reading the table."""

UNSETTLED: frozenset[str] = frozenset({"queued", "running"})
"""The states that still owe work — what an owner going away has to stop.

Beside the vocabulary it reads, for `SETTLED_STATUSES`' reason one seam over: a
consumer that spelled its own copy got it wrong the moment a state was added."""

Slot: TypeAlias = tuple[str, int]
"""`(key, limit)`: which queue this job waits in, and how many of that queue run
at once. The key is the producer's to choose — a parent session id bounds one
parent's fan-out.

A deployment-wide bound is *not* spelled here: it is `Config.concurrency`, set by
whoever runs the harness rather than by the row that happens to start the work."""

_Key: TypeAlias = tuple[str, str, str]
"""A queue's identity: which kind of bound, for which job kind, and whose.

Three parts rather than two because the *kind* has to be in there. A producer
picks its own key and the obvious one is a session id — which two producers can
reach for at once, silently sharing a queue whose limit is whichever of them
arrived first. Keyed by kind as well, that cannot happen, and no producer has to
know the others exist to avoid it."""

CHILDREN_KIND = "subagent"
"""The job kind a subagent drive runs under, and the one a deployment usually caps.

**Named here so a host can spell the cap without importing a bundle**, which is
the whole of what this constant is for — `ph daemon --max-concurrent-children`
needs a key, and reaching into `ph-rlm` for it from `ph-app` would be worse than
one string in the seam that defines what a job kind is. The *number* is not here:
see `Config.concurrency`."""

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

    waiting: anyio.CancelScope | None = None
    """The wait for a slot, while there is one. Internal to `JobService`.

    A body parked on a limiter is not running, so it reads no cancel token —
    cooperative cancellation cannot reach it. This is what `cancel` cancels
    instead, and it is why a queued job can be stopped at all."""

    def cancel(self) -> None:
        """Stop this job, whether it is running or still waiting for a slot."""
        self.token.cancel("job cancelled")
        if self.waiting is not None:
            self.waiting.cancel()


@dataclass(slots=True)
class _Queue:
    """One slot queue, and how many jobs still hold a place in it."""

    key: _Key
    """Its own identity, so a job holding one needs no parallel list of keys to
    give its place back."""
    limiter: anyio.CapacityLimiter
    held: int = 0
    """Admitted jobs that have not settled — queued and running alike.

    A refcount rather than a "was that the last one?" scan of the job table: a
    settled job may legitimately stay in that table (`forget` is the owner's
    call, not this seam's), so the scan would answer *no* forever and the queue
    would outlive every producer that ever used it."""


@dataclass(slots=True)
class JobService:
    """The service published as `ctx.jobs`."""

    ctx: Context
    caps: Mapping[str, int] = field(default_factory=dict)
    """Deployment-wide ceilings by job kind — `Config.concurrency`.

    Read here rather than passed at each `start`, because it is a fact about the
    deployment and not about any one piece of work: a producer that had to quote
    it would be a producer that could forget to."""
    _jobs: dict[str, Job] = field(default_factory=dict)
    _queues: dict[_Key, _Queue] = field(default_factory=dict)
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
        slot: Slot | None = None,
        on_queued: Callable[[], None] | None = None,
    ) -> Job:
        """Start one background job, owned by `scope` (default: this seam's).

        `scope` is what bounds the job's lifetime: the delegation, the session,
        the process. Disposing it abandons the job — cancelled if still running,
        and dropped from the table either way.

        **`slot=(key, limit)` queues instead of refusing.** `start` still returns
        at once and the handle is real — what waits is the *body*, so a producer's
        admission, its record and its identity are untouched by how busy it is. A
        job waiting for a slot is `queued`; it takes one in admission order, and
        frees it on every ending — done, failed or cancelled — so one failure
        cannot wedge the queue behind it. The first caller of a key fixes that
        key's limit; a later `(key, other)` joins the existing queue rather than
        resizing it, because two producers disagreeing about a bound is not
        something this seam can settle for them.

        `on_queued` fires only when there is actually a wait, so a producer can
        record *that* rather than infer it. A parameter and not a field on `Job`
        for a stated reason: a callable on a registry's value is a row body the
        ownership walk has to classify, and this one is the caller's own, handed
        back to the caller, rather than a registration.
        """
        job = Job(id=f"{kind}-{secrets.token_hex(4)}", kind=kind, label=label, token=CancelToken())
        waits = self._waits(kind, slot)
        if waits:
            job.state = "queued"
        self._jobs[job.id] = job

        def enter() -> Disposer:
            def abandon() -> None:
                # The entry's presence *is* the "still owned" flag: `forget` pops
                # it before deregistering, so an owner that released a finished
                # job cannot have it reported as cancelled here.
                if self._jobs.pop(job.id, None) is None:
                    return
                # **`queued` counts as owed, not just `running`.** A job still
                # waiting for a slot has a parked body and, once it is out of
                # this table, no owner left to stop it: it would take a slot
                # nobody wants, run work whose owner is gone, and hold
                # `ctx.drain()` open until it did. This read `== "running"` while
                # the only states were running-or-settled, and a whole kind
                # became `queued` the day a deployment cap shipped.
                if job.state in UNSETTLED:
                    job.cancel()

            return abandon

        async def body() -> None:
            holds = False
            try:
                if waits:
                    holds = await self._take(job, waits, on_queued)
                    if not holds:
                        # Cancelled where it waited. It never ran, which is what
                        # `cancelled` says and why the work below is not entered.
                        job.state = "cancelled"
                        return
                if job.token.cancelled:
                    job.state = "cancelled"
                    return
                job.state = "running"
                job.result = await maybe_await(run(job))
                job.state = "cancelled" if job.token.cancelled else "done"
            except Exception as error:
                job.state = "failed"
                job.error = error
                log.debug("ph.seams.jobs: job %s failed", job.id, exc_info=True)
            finally:
                if holds:
                    for queue in waits:
                        queue.limiter.release()
                self._leave(waits)
                self.ctx.emit("job/settled", job, contained=True)

        try:
            job.release = await self.ctx.owner_for(scope).effect(enter, label=f"job({job.id})")
            self.ctx.emit("job/started", job, contained=True)
            if self._scope is not None:
                self._scope.start_soon(body)
            else:
                self.ctx.detach(body(), label=f"job {job.id}")
        except BaseException:
            # **A job that never reaches its body gives its places back here, or
            # nothing ever does.** `body`'s `finally` is the only other release,
            # so a disposed owner or a closed task group between the two used to
            # strand one refcount per attempt — and a queue whose count never
            # returns to zero is one this table keeps for the life of the
            # process, keyed by whatever the producer chose. Usually a session id.
            self._leave(waits)
            self._jobs.pop(job.id, None)
            raise
        return job

    def _waits(self, kind: str, slot: Slot | None) -> Sequence[_Queue]:
        """Every queue this job waits in, **narrowest first**.

        The order is the whole of why holding two is safe. A job takes its
        producer's slot before the deployment's, so it only occupies global
        capacity once it is genuinely next for its own parent; the other way
        round, one parent's queued children would sit on every deployment slot
        while waiting for a bound of their own, starving every other parent. No
        cycle is possible either, because every job acquires in this same order.
        """
        waits: list[_Queue] = []
        if slot is not None:
            waits.append(self._queue(("slot", kind, slot[0]), slot[1]))
        cap = self.caps.get(kind)
        if cap is not None:
            waits.append(self._queue(("kind", kind, ""), cap))
        # Counted here, where the places are taken, so the increment and the
        # `_leave` that undoes it cannot be one caller apart.
        for queue in waits:
            queue.held += 1
        return waits

    def _queue(self, key: _Key, limit: int) -> _Queue:
        """This key's queue, made on first use."""
        queue = self._queues.get(key)
        if queue is None:
            queue = self._queues[key] = _Queue(key=key, limiter=anyio.CapacityLimiter(limit))
        return queue

    async def _take(
        self, job: Job, waits: Sequence[_Queue], on_queued: Callable[[], None] | None
    ) -> bool:
        """Wait for a place in every queue. `False` if cancelled while waiting.

        The wait is its own cancel scope, not the body's: a job cancelled while
        queued must stop waiting, while one cancelled while *running* is stopped
        through its token — and a single scope over both would cancel the work
        mid-flight, which is the opposite of cooperative.

        A cancel releases whatever prefix it had taken here, so the caller's
        `finally` has one rule — release everything, or nothing — rather than a
        list of what happened to be acquired.
        """
        if on_queued is not None and any(queue.limiter.available_tokens < 1 for queue in waits):
            on_queued()
        taken: list[_Queue] = []
        with anyio.CancelScope() as waiting:
            job.waiting = waiting
            for queue in waits:
                await queue.limiter.acquire()
                taken.append(queue)
        job.waiting = None
        if not waiting.cancelled_caught:
            return True
        for queue in taken:
            queue.limiter.release()
        return False

    def _leave(self, waits: Sequence[_Queue]) -> None:
        """Drop this job's places, and each queue once nobody holds one."""
        for queue in waits:
            queue.held -= 1
            if queue.held <= 0:
                self._queues.pop(queue.key, None)

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


class Config(WireModel):
    """Row config for the local job runner."""

    concurrency: dict[str, int] = Field(default_factory=dict)
    """How many jobs of each kind this deployment runs at once; the rest queue.

    A **deployment's** bound, which is why it is here and not an argument every
    producer passes: it answers "what can this host carry", where a producer's
    own `slot=` answers "what is one caller's fair share". Both apply, narrowest
    first. A kind that is absent is uncapped.

    **Empty by default, and the number lives with the producer.** A default here
    naming `subagent` would be this seam — which treats `kind` as a free string
    everywhere else — knowing one bundle's vocabulary, and `ph-base` ships no
    subagent provider for it to bound. The `rlm` bundle, which does ship one,
    sets its own figure beside the per-parent one, so the two numbers an operator
    compares sit in one file."""


@plugin("jobs-local", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the local job runner."""
    ctx.provide(JOBS, JobService(ctx=ctx, caps=dict(config.concurrency)))
