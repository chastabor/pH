"""`llm-retry`: bounded retries, and the one failure that it deliberately does not retry.

Retrying can go wrong in two directions. A harness that retries nothing is
fragile against ordinary rate limits. A harness that retries everything turns a
context-window overflow into an infinite loop that is billed for every attempt.

So the classification is explicit:

* **Transient** (rate limits, 5xx, timeouts, empty responses): retry with
  exponential backoff. When the provider sent its own `retry_after`, the retry
  honors it, because the provider knows better than the backoff curve does.
* **`FILE_EXPIRED`**: retry. It is the clearest case in the list. The adapter
  has already dropped the dead handle, so the second attempt is against a
  freshly uploaded file and is not the same request twice.
* **`CONTEXT_WINDOW_EXCEEDED`**: **do not retry**. The request cannot fit, and
  it will not fit on the second attempt either. Compaction acts on this signal
  (G4, Phase 4). Consuming it here would hide the one error that has a real
  remedy.
* **Everything else**: do not retry. An unknown failure that is retried is an
  unknown failure that is billed twice.

@module ph.llm.retry
"""

from __future__ import annotations

import logging

from ..agent.types import RequestErrorAction, RequestFailure
from ..cordis import Context, Next, plugin
from ..keys import SESSIONS
from ..session.writers import log_writer
from ..wire import WireModel
from .types import CONTEXT_WINDOW_EXCEEDED, EMPTY_RESPONSE, FILE_EXPIRED, TIMEOUT, LlmFailure

_LOG = log_writer(__name__)

__all__ = ["RETRIED", "ROW", "TRANSIENT_CODES", "apply", "is_transient"]

log = logging.getLogger("ph.llm.retry")

RETRIED = "llm/retry"
"""The attempt record: what was retried, when and why. It is an audit trail. The
count that the budget spends belongs to the loop, on `RequestFailure.retries_by`."""

ROW = "llm-retry"
"""The id of this row, and the key that its own retries are counted under."""

TRANSIENT_CODES: frozenset[str] = frozenset(
    {
        "RATE_LIMIT",
        "RATE_LIMITED",
        "SERVER_ERROR",
        "SERVICE_UNAVAILABLE",
        TIMEOUT,
        "CONNECTION_ERROR",
        "OVERLOADED",
        EMPTY_RESPONSE,
        # Transient in the strict sense that this module means: the state that
        # caused the failure is already gone. An adapter that raises this code
        # has invalidated the handle first (P7-03). The retry therefore builds
        # the request again against a fresh upload, and does not repeat the
        # request that failed.
        FILE_EXPIRED,
    }
)

_TRANSIENT_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def is_transient(failure: LlmFailure) -> bool:
    """Whether this failure is worth a second attempt."""
    if failure.code == CONTEXT_WINDOW_EXCEEDED:
        # Never. The remedy is compaction, and consuming the signal here would
        # take that remedy away.
        return False
    if failure.code in TRANSIENT_CODES:
        return True
    return failure.status in _TRANSIENT_STATUS


class Config(WireModel):
    """Row config for the retry policy."""

    max_attempts: int = 3
    base_delay_ms: int = 500
    max_delay_ms: int = 20_000
    count_all_retries: bool = False
    """Whether the retries of every row spend the budget of this row, or only its
    own retries do (G13).

    The default is off, which is what this row always did. Only its own retries
    count, so a step that was compacted and retried after an overflow still has
    every transient retry that it was configured for. When the setting is on, a
    compaction retry, or a retry by any other row, leaves one retry fewer. That
    gives a hard ceiling on model calls for each step. In both cases the number
    belongs to the loop (`RequestFailure.retries_by`, attributed by cordis).
    This row holds its opt-in and nothing else."""


@plugin(ROW, affects="environment", config=Config, inject=[SESSIONS])
async def apply(ctx: Context, config: Config) -> None:
    """Retry transient request failures with bounded backoff."""
    settings = config

    async def on_error(
        failure_payload: RequestFailure,
        next_: Next[RequestErrorAction | None],
    ) -> RequestErrorAction | None:
        failure = failure_payload.failure
        if not is_transient(failure):
            return await next_()
        session = failure_payload.session
        seen = (
            failure_payload.retries
            if settings.count_all_retries
            else failure_payload.retries_by.get(ROW, 0)
        )
        if seen + 1 >= settings.max_attempts:
            log.debug("ph.llm.retry: giving up on %s after %s attempts", failure.code, seen + 1)
            return await next_()

        delay_ms = min(settings.base_delay_ms * (2**seen), settings.max_delay_ms)
        if failure.provider_retry_after_ms is not None:
            # The number of the provider wins over our curve, because the
            # provider knows when the bucket refills.
            delay_ms = max(delay_ms, failure.provider_retry_after_ms)

        account = {
            "turn": failure_payload.turn,
            "step": failure_payload.step,
            "attempt": seen + 1,
            "delayMs": delay_ms,
            "code": failure.code,
        }

        def record() -> None:
            _LOG.append(session, RETRIED, account)

        return RequestErrorAction(kind="retry", delay_ms=delay_ms, granted=record)

    ctx.on("agent/request-error", on_error)
