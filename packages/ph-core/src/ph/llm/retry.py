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

import anyio

from ..agent.types import RequestErrorAction, RequestFailure
from ..cordis import Context, Next, plugin
from ..json import as_int
from ..keys import SESSIONS
from ..session import Session
from ..wire import WireModel
from .types import CONTEXT_WINDOW_EXCEEDED, EMPTY_RESPONSE, FILE_EXPIRED, LlmFailure

__all__ = ["RETRIED", "TRANSIENT_CODES", "apply", "attempts_so_far", "is_transient"]

log = logging.getLogger("ph.llm.retry")

RETRIED = "llm/retry"
"""The attempt record, which is also where the attempt *count* is read from."""

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


def attempts_so_far(session: Session, turn: int, step: int) -> int:
    """How many times this step has already been retried, read from the log.

    **The count is derived, not held** (I4). It was a dict on the row, keyed by
    `turn:step` — and a row is mounted once per root while every agent beneath it
    dispatches to the same listener, so two subagents at the same coordinates
    shared one budget. The entry was also dropped only on give-up, so a step that
    succeeded on its second attempt left its count behind for whoever reached
    `1:1` next. A sibling could therefore be refused every retry it had, which is
    the case this exists for.

    `llm/retry` already records each attempt, so the log is the count. The latest
    one is enough: a step's retries are consecutive by construction — one agent
    drives one step at a time, and the next append for a different step is the
    one after that — so an event naming other coordinates means this step has had
    none yet.
    """
    latest = session.latest(RETRIED)
    if latest is None:
        return 0
    if as_int(latest.data.get("turn")) != turn or as_int(latest.data.get("step")) != step:
        return 0
    return as_int(latest.data.get("attempt"))


@plugin("llm-retry", config=Config, inject=[SESSIONS])
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
        seen = attempts_so_far(session, failure_payload.turn, failure_payload.step)
        if seen + 1 >= settings.max_attempts:
            log.debug("ph.llm.retry: giving up on %s after %s attempts", failure.code, seen + 1)
            return await next_()

        delay_ms = min(settings.base_delay_ms * (2**seen), settings.max_delay_ms)
        if failure.provider_retry_after_ms is not None:
            # The provider's own number beats our curve: it knows when the
            # bucket refills.
            delay_ms = max(delay_ms, failure.provider_retry_after_ms)

        session.append(
            RETRIED,
            {
                "turn": failure_payload.turn,
                "step": failure_payload.step,
                "attempt": seen + 1,
                "delayMs": delay_ms,
                "code": failure.code,
            },
        )
        await anyio.sleep(delay_ms / 1000)
        return RequestErrorAction(kind="retry", delay_ms=delay_ms)

    ctx.on("agent/request-error", on_error)
