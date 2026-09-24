"""`llm-retry` — bounded retries, and the one failure it deliberately declines.

Retrying is easy to get wrong in two directions. Retrying nothing makes a
harness fragile against ordinary rate limits; retrying everything turns a
context-window overflow into an infinite loop that bills for every attempt.

So the classification is explicit:

* **transient** (rate limits, 5xx, timeouts, empty responses) → retry with
  exponential backoff, honoring a provider's own `retry_after` when it sent one,
  because the provider knows better than the backoff curve does;
* **`FILE_EXPIRED`** → retry, and it is the clearest case in the list: the
  adapter has already dropped the dead handle, so the second attempt is against
  a freshly uploaded file rather than the same request twice;
* **`CONTEXT_WINDOW_EXCEEDED`** → **do not retry**. The request cannot fit, and
  it will not fit on the second attempt either. This is the signal compaction
  keys off (G4, Phase 4); consuming it here would hide the one error that has a
  real remedy;
* **everything else** → do not retry. An unknown failure retried is an unknown
  failure billed twice.

@module ph.llm.retry
"""

from __future__ import annotations

import logging

from ..agent.types import RequestErrorAction, RequestFailure
from ..cordis import Context, Next, plugin
from ..keys import SESSIONS
from ..session.writers import log_writer
from ..wire import WireModel
from .types import CONTEXT_WINDOW_EXCEEDED, EMPTY_RESPONSE, FILE_EXPIRED, LlmFailure

_LOG = log_writer(__name__)

__all__ = ["RETRIED", "ROW", "TRANSIENT_CODES", "apply", "is_transient"]

log = logging.getLogger("ph.llm.retry")

RETRIED = "llm/retry"
"""The attempt record: what was retried, when and why. An audit trail — the count
the budget spends is the loop's, on `RequestFailure.retries_by`."""

ROW = "llm-retry"
"""This row's id, and the key its own retries are counted under."""

TRANSIENT_CODES: frozenset[str] = frozenset(
    {
        "RATE_LIMIT",
        "RATE_LIMITED",
        "SERVER_ERROR",
        "SERVICE_UNAVAILABLE",
        "TIMEOUT",
        "CONNECTION_ERROR",
        "OVERLOADED",
        EMPTY_RESPONSE,
        # Transient in the strict sense this module means: the state that caused
        # it is already gone. An adapter raising this has invalidated the handle
        # first (P7-03), so the retry rebuilds the request against a fresh
        # upload rather than repeating the one that failed.
        FILE_EXPIRED,
    }
)

_TRANSIENT_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def is_transient(failure: LlmFailure) -> bool:
    """Whether this failure is worth a second attempt."""
    if failure.code == CONTEXT_WINDOW_EXCEEDED:
        # Never: the remedy is compaction, and swallowing the signal here would
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
    """Whether every row's retries spend this row's budget, or only its own (G13).

    Off by default, which is what this row always did: only its own retries
    count, so a step that was compacted and retried after an overflow still has
    every transient retry it was configured for. On, a compaction retry — or any
    other row's — is one fewer, for a hard ceiling on model calls per step.
    Either way the number is the loop's (`RequestFailure.retries_by`, attributed
    by cordis); this row holds its opt-in and nothing else."""


@plugin(ROW, config=Config, inject=[SESSIONS])
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
            # The provider's own number beats our curve: it knows when the
            # bucket refills.
            delay_ms = max(delay_ms, failure.provider_retry_after_ms)

        _LOG.append(
            session,
            RETRIED,
            {
                "turn": failure_payload.turn,
                "step": failure_payload.step,
                "attempt": seen + 1,
                "delayMs": delay_ms,
                "code": failure.code,
            },
        )
        return RequestErrorAction(kind="retry", delay_ms=delay_ms)

    ctx.on("agent/request-error", on_error)
