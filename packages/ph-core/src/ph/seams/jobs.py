"""`ctx.jobs`: background work with a handle, a cancel and a completion.

A job is long-running work that is not a turn. The producers in this tree are a
`/refine` planner pass, the drive of a subagent, and a background run of a
`ph-utcp` tool. A job is deliberately *not* a tool call. It outlives the step
that started it, so it needs its own identity and its own cancellation, and it
cannot borrow those of the turn.

**A job is an effect of the scope that owns it (I2).** It outlives the *step*,
not the agent or the session. The drive job of a subagent belongs to the
delegation. A `/refine` pass belongs to the agent that asked. A background run
belongs to the agent that owns the tree. So `start` takes a `scope=`, as every
other registration here does, and the dispose of that scope cancels the job and
drops its entry. Without an owner the table only grew, and its bound would have
been a number that somebody picked.

"This job is over" has two halves, and they are deliberately distinct:

* **abandoned**: the owning scope went away while the work still ran. The job
  is canceled and then forgotten.
* **released** (`forget`): the owner knows that the work is finished and wants
  the entry gone. The job is forgotten, and nothing is canceled. Without this
  half, a job whose own body triggers the teardown of its owner would report
  `canceled` for work that completed.

Cancellation is cooperative. `Job.cancel()` sets a token, and a body that never
reads the token runs to completion. It also runs past a teardown: `ctx.drain()`
does not wait for jobs, because their owners stop them in `dispose`. A caller
that wants to wait calls `settled()`.

**A producer bounds its own concurrency with a slot** (`slot=`). Work that
arrives faster than it should run is queued and not refused. The caller asked
for all of it, and a refusal would treat a question about resources as a
question about intent. The primitive is here and not in each producer, because
"wait for a free slot" is forty lines of subtle async: a cancel scope around the
wait, a release on every ending, and a table that must be dropped. A producer
that wrote those lines itself would have to find each of those details again.
Every producer queues the same way, and a second provider inherits the queue
and does not derive it again.

@module ph.seams.jobs

"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import anyio
from pydantic import Field

from ..cancel import CancelToken
from ..cordis import Context, Disposer, Releaser, events, maybe_await, plugin
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

JobState: TypeAlias = Literal["queued", "running", "done", "failed", "canceled"]
"""`queued` means that the job is admitted and waits for a slot. See the `slot=`
of `JobService.start`.

It is a state and not an absence, because "not started yet" and "not started at
all" are different things to a reader of the table."""

UNSETTLED: frozenset[str] = frozenset({"queued", "running"})
"""The states that still owe work. These are what a departing owner must stop.

The set is defined beside the vocabulary that it reads, for the same reason as
`SETTLED_STATUSES` in `ph.seams.subagents`: a consumer that wrote its own copy
had it wrong as soon as a state was added."""

Slot: TypeAlias = tuple[str, int]
"""`(key, limit)`: the queue that this job waits in, and how many jobs of that
queue run at one time. The producer chooses the key. A parent session id as the
key bounds the fan-out of one parent.

A deployment-wide bound is *not* written here. It is `Config.concurrency`, set
by whoever runs the harness and not by the row that happens to start the work."""

_Key: TypeAlias = tuple[str, str, str]
"""The identity of a queue: which kind of bound, for which job kind, and whose.

It has three parts and not two, because the *kind* must be in it. A producer
picks its own key, and the obvious key is a session id. Two producers can pick
the same session id. They would then silently share a queue whose limit is that
of whichever producer arrived first. With the kind in the key that cannot
happen, and no producer has to know that the others exist to avoid it."""

CHILDREN_KIND = "subagent"
"""The job kind that the drive of a subagent runs under, and the kind that a
deployment usually caps.

**It is named here so that a host can write the cap without importing a
bundle.** That is the only purpose of this constant.
`phern daemon --max-concurrent-children` needs a key. One string in the seam
that defines what a job kind is, is better than an import of `ph-rlm` into
`ph_app` for that key. The *number* is not here: see `Config.concurrency`."""

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
    result: object = None
    error: BaseException | None = None
    release: Releaser | None = None
    """The stop of this job on its owning scope. It cancels the job if the scope
    goes first. `start` sets it, and `JobService.forget` drops it without running
    it. A caller does not drop it directly, because the entry and the stop must
    be removed together."""

    waiting: anyio.CancelScope | None = None
    """The wait for a slot, while there is one. It is internal to `JobService`.

    A body that waits on a limiter is not running, so it reads no cancel token,
    and cooperative cancellation cannot reach it. `cancel` cancels this scope
    instead. That is why a queued job can be stopped at all."""

    def cancel(self) -> None:
        """Stop this job, whether it is running or still waiting for a slot."""
        self.token.cancel("job canceled")
        if self.waiting is not None:
            self.waiting.cancel()


@dataclass(slots=True)
class _Queue:
    """One slot queue, and how many jobs still hold a place in it."""

    key: _Key
    """The identity of the queue. A job that holds a place then needs no parallel
    list of keys to give the place back."""
    limiter: anyio.CapacityLimiter
    held: int = 0
    """The admitted jobs that have not settled, queued and running alike.

    It is a reference count and not a scan of the job table for "was that the
    last one?". A settled job may legitimately stay in that table, because
    `forget` is the decision of the owner and not of this seam. The scan would
    then answer *no* forever, and the queue would outlive every producer that
    ever used it."""


@dataclass(slots=True)
class JobService:
    """The service published as `ctx.jobs`."""

    ctx: Context
    caps: Mapping[str, int] = field(default_factory=dict)
    """The deployment-wide ceilings by job kind (`Config.concurrency`).

    They are read here and not passed at each `start`, because a ceiling is a
    fact about the deployment and not about one piece of work. A producer that
    had to state it could forget to."""
    _jobs: dict[str, Job] = field(default_factory=dict)
    _queues: dict[_Key, _Queue] = field(default_factory=dict)
    _bodies: int = 0
    """The job bodies that were spawned and have not ended, queued ones included.
    This is what `settled` waits for. It is a count and not the table, because
    the table drops an abandoned job while its body may still be ending."""
    _quiet: anyio.Event | None = None
    """Set when `_bodies` reaches zero, for whoever is in `settled` then."""

    @property
    def in_flight(self) -> int:
        """Job bodies started here and not yet ended, queued ones included."""
        return self._bodies

    async def settled(self) -> None:
        """Wait until every job body started here has ended.

        This is for a caller that chooses to wait. No teardown does:
        `ctx.drain()` leaves jobs to their owners. The wait has no bound,
        because the caller decides how long a job may take. A body that never
        reads its token holds this wait until the body finishes.
        """
        while self._bodies:
            if self._quiet is None:
                self._quiet = anyio.Event()
            await self._quiet.wait()

    def _ended(self) -> None:
        """Count one body as ended. Wake whoever is in `settled` when none are left."""
        self._bodies -= 1
        if not self._bodies and self._quiet is not None:
            self._quiet.set()
            self._quiet = None

    async def start(
        self,
        *,
        kind: str,
        label: str,
        run: Callable[[Job], object],
        scope: Context | None = None,
        slot: Slot | None = None,
        on_queued: Callable[[], None] | None = None,
        on_settled: Callable[[Job], None] | None = None,
    ) -> Job:
        """Start one background job, owned by `scope` (default: the scope of this
        seam).

        `scope` bounds the lifetime of the job: the delegation, the session, the
        process. The dispose of `scope` abandons the job. The job is canceled if
        it has not settled, and it is dropped from the table in either case.

        **`slot=(key, limit)` queues and does not refuse.** `start` still returns
        at once and the handle is real. Only the *body* waits, so the admission,
        the record and the identity of a producer do not depend on how busy it
        is. A job that waits for a slot is `queued`. It takes a slot in
        admission order, and frees the slot on every ending (done, failed or
        canceled), so one failure cannot block the queue behind it. The first
        caller of a key fixes the limit of that key. A later `(key, other)`
        joins the existing queue and does not resize it, because this seam
        cannot settle a disagreement between two producers about a bound.

        `on_queued` fires only when the job actually waits, so a producer can
        record *that* and need not infer it. It is a parameter and not a field
        on `Job` for a stated reason. A callable on the value of a registry is a
        row body that the ownership walk has to classify. This callable belongs
        to the caller and is handed back to the caller. It is not a
        registration.

        **`on_settled` is called however the job ends** (S6): done, failed, or
        canceled while it ran. It is also called for a job that was canceled
        before it ran, either while it waited for a slot or before its first
        step. Such a job never enters `run`. A producer uses `on_settled` to
        take down what it put up for the job. When that cleanup was inside
        `run`, an entry outlived every job that never got there. `on_settled`
        is called before `job/settled` is announced. It is not called if `start`
        raises, because the caller sees that.
        """
        job = Job(id=f"{kind}-{secrets.token_hex(4)}", kind=kind, label=label, token=CancelToken())
        waits = self._waits(kind, slot)
        if waits:
            job.state = "queued"
        self._jobs[job.id] = job

        def enter() -> Disposer:
            def abandon() -> None:
                # This never runs for a job that its owner released: `forget`
                # drops this disposer without running it.
                del self._jobs[job.id]
                # **`queued` counts as owed, and not only `running`.** A job that
                # still waits for a slot has a waiting body. When the job is out
                # of this table, no owner is left to stop it. It would take a
                # slot that nobody wants, run work whose owner is gone, and hold
                # `settled()` open until it finished. This test read
                # `== "running"` while the only states were running or settled,
                # and a whole kind became `queued` on the day that a deployment
                # cap shipped.
                if job.state in UNSETTLED:
                    job.cancel()

            return abandon

        async def body() -> None:
            holds = False
            try:
                if waits:
                    holds = await self._take(job, waits, on_queued)
                    if not holds:
                        # The job was canceled where it waited. It never ran.
                        # `canceled` says that, and it is why the work below is
                        # not entered.
                        job.state = "canceled"
                        return
                if job.token.canceled:
                    job.state = "canceled"
                    return
                job.state = "running"
                job.result = await maybe_await(run(job))
                job.state = "canceled" if job.token.canceled else "done"
            except Exception as error:
                job.state = "failed"
                job.error = error
                log.debug("ph.seams.jobs: job %s failed", job.id, exc_info=True)
            finally:
                # This comes first, so that nothing below can strand the count.
                # Nothing below awaits either, so a waiter in `settled` wakes
                # after all of it.
                self._ended()
                if job.state == "running":
                    # **Derived, not caught** (L4). `except Exception` misses a
                    # `BaseException`: a foreign cancel, a `BaseExceptionGroup`
                    # from a task group, `KeyboardInterrupt`. The state then
                    # stayed `"running"`, and `job/settled` announced a job as
                    # still running while it went away. The `job.token.canceled`
                    # check above covers only a cancel through the *own* token
                    # of the job.
                    #
                    # The state is settled here and not by a third `except`, for
                    # the reason that `_turn_reason` (`ph.agent_loop.driver`)
                    # gives. Every ordinary exit assigns a state first, so
                    # `"running"` here cannot be a job that finished. A branch
                    # for each exception type is a list that the next base
                    # exception is not on.
                    job.state = "canceled"
                if holds:
                    for queue in waits:
                        queue.limiter.release()
                self._leave(waits)
                if on_settled is not None:
                    try:
                        on_settled(job)
                    except Exception:
                        log.exception("ph.seams.jobs: settling job %s raised", job.id)
                self.ctx.emit("job/settled", job)

        try:
            job.release = await self.ctx.owner_for(scope).effect(enter, label=f"job({job.id})")
            self.ctx.emit("job/started", job)
            # Every body runs on `ctx.detach` with `leave`, so `start` returns
            # before the work does. A job that outlives the step that started it
            # is the purpose of the seam. A teardown stops the job through its
            # owner and does not wait for it. Nothing is dropped, because the
            # stop of the owner keeps the work: a child is suspended and resumed
            # later, and a refinement is considered again.
            #
            # `leave`: the `effect` above is the stop of this job. It is on the
            # owner of the job, and it cancels the token of the job. A cancel
            # from the scope of this seam would be a second stop that arrives
            # before the stop of the owner, and this row unwinds before the
            # agents that own its jobs. `leave` also keeps the job out of what
            # `drain` waits for (`AtDispose`).
            self.ctx.detach(body(), label=f"job {job.id}", at_dispose="leave")
            # The body is counted when it is sure to run, and before it can run:
            # neither spawn runs the body before this line.
            self._bodies += 1
        except BaseException:
            # **A job that never reaches its body gives its places back here, or
            # nothing ever does.** The `finally` of `body` is the only other
            # release. A disposed owner between the two used to strand one
            # reference count for each attempt. A queue whose
            # count never returns to zero stays in this table for the life of
            # the process, under whatever key the producer chose. That key is
            # usually a session id.
            self._leave(waits)
            self.forget(job.id)
            raise
        return job

    def _waits(self, kind: str, slot: Slot | None) -> Sequence[_Queue]:
        """Every queue that this job waits in, **narrowest first**.

        The order is the reason why it is safe to hold two places. A job takes
        the slot of its producer before the slot of the deployment, so it
        occupies global capacity only when it is next for its own parent. In
        the other order, the queued children of one parent would hold every
        deployment slot while they waited for the bound of that parent, and
        every other parent would starve. No cycle is possible either, because
        every job acquires in this same order.
        """
        waits: list[_Queue] = []
        if slot is not None:
            waits.append(self._queue(("slot", kind, slot[0]), slot[1]))
        cap = self.caps.get(kind)
        if cap is not None:
            waits.append(self._queue(("kind", kind, ""), cap))
        # The places are counted here, where they are taken. The increment and
        # the `_leave` that undoes it then cannot be one caller apart.
        for queue in waits:
            queue.held += 1
        return waits

    def _queue(self, key: _Key, limit: int) -> _Queue:
        """The queue of this key, made on first use."""
        queue = self._queues.get(key)
        if queue is None:
            queue = self._queues[key] = _Queue(key=key, limiter=anyio.CapacityLimiter(limit))
        return queue

    async def _take(
        self, job: Job, waits: Sequence[_Queue], on_queued: Callable[[], None] | None
    ) -> bool:
        """Wait for a place in every queue. Returns `False` if the job was canceled
        while it waited.

        The wait has its own cancel scope, which is not the scope of the body. A
        job canceled while it is queued must stop waiting. A job canceled while
        it is *running* is stopped through its token. One scope over both would
        cancel the work in mid-flight, which is the opposite of cooperative.

        A cancel releases the places that the job had already taken here. The
        `finally` of the caller therefore has one rule, release everything or
        nothing, and needs no list of what was acquired.
        """
        if job.token.canceled:
            # The job was canceled before it began to wait. There is nothing to
            # announce and nothing taken. A call to `on_queued` here made a
            # producer record a wait for work that would never run.
            return False
        if on_queued is not None and any(queue.limiter.available_tokens < 1 for queue in waits):
            on_queued()
        taken: list[_Queue] = []
        holds = False
        try:
            with anyio.CancelScope() as waiting:
                job.waiting = waiting
                for queue in waits:
                    await queue.limiter.acquire()
                    taken.append(queue)
            holds = not waiting.cancelled_caught
        finally:
            job.waiting = None
            # Whatever ended the wait gives back what the job had taken. The
            # ending may be the own cancel of the job, or a stop from outside,
            # such as the cancel of the task that runs the body when the event
            # loop shuts down. When only the own cancel released, an outside
            # stop kept the place in the narrower queue, and a job that still
            # waited in that queue waited forever.
            if not holds:
                for queue in taken:
                    queue.limiter.release()
        return holds

    def _leave(self, waits: Sequence[_Queue]) -> None:
        """Drop the places of this job, and each queue when no job holds a place in it."""
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
        """Drop the entry of a finished job without canceling it.

        This is for an owner that knows that the work is done, such as a
        delegation whose child has settled. The table then does not hold one
        entry for each unit of past work for the life of the scope. `forget`
        also drops the stop of the job from the scope, without running it, so
        that the scope holds no teardown for work that is already over.
        """
        job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        if job.release is not None:
            job.release.drop()
        return True


class Config(WireModel):
    """Row config for the local job runner."""

    concurrency: dict[str, int] = Field(default_factory=dict)
    """How many jobs of each kind this deployment runs at one time. The rest
    queue.

    This is the bound of the **deployment**, which is why it is here and is not
    an argument that every producer passes. It answers "what can this host
    carry". The `slot=` of a producer answers "what is the fair share of one
    caller". Both apply, narrowest first. A kind that is absent has no cap.

    **It is empty by default, and the bundle of the producer sets the number.**
    A default here that named `subagent` would make this seam know the
    vocabulary of one bundle, and the seam treats `kind` as a free string
    everywhere else. `ph-base` also ships no subagent provider for such a
    default to bound. The `rlm` bundle does ship one. It sets its own figure
    beside the per-parent figure, so that the two numbers that an operator
    compares are in one file."""


@plugin("jobs-local", affects="deployment", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the local job runner."""
    ctx.provide(JOBS, JobService(ctx=ctx, caps=dict(config.concurrency)))
