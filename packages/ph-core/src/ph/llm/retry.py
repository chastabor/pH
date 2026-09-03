"""`llm-retry` — bounded retries, and the one failure it deliberately declines.

Retrying is easy to get wrong in two directions. Retrying nothing makes a
harness fragile against ordinary rate limits; retrying everything turns a
context-window overflow into an infinite loop that bills for every attempt.

So the classification is explicit:

* **transient** (rate limits, 5xx, timeouts, empty responses) → retry with
  exponential backoff, honouring a provider's own `retry_after` when it sent one,
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
from collections.abc import Callable
from typing import Any

import anyio

from ..cordis import Context, plugin
from ..wire import WireModel
from .types import CONTEXT_WINDOW_EXCEEDED, EMPTY_RESPONSE, FILE_EXPIRED, LlmFailure

__all__ = ["TRANSIENT_CODES", "apply", "is_transient"]

log = logging.getLogger("ph.llm.retry")

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


@plugin("llm-retry", config=Config, inject=["sessions"])
async def apply(ctx: Context, config: Config) -> None:
    """Retry transient request failures with bounded backoff."""
    settings = config
    attempts: dict[str, int] = {}

    async def on_error(failure_payload: Any, next_: Callable[..., Any]) -> Any:
        from ..agent.types import RequestErrorAction

        failure: LlmFailure = failure_payload.failure
        key = f"{failure_payload.turn}:{failure_payload.step}"
        if not is_transient(failure):
            attempts.pop(key, None)
            return await next_()
        seen = attempts.get(key, 0)
        if seen + 1 >= settings.max_attempts:
            log.debug("ph.llm.retry: giving up on %s after %s attempts", failure.code, seen + 1)
            attempts.pop(key, None)
            return await next_()
        attempts[key] = seen + 1

        delay_ms = min(settings.base_delay_ms * (2**seen), settings.max_delay_ms)
        if failure.provider_retry_after_ms is not None:
            # The provider's own number beats our curve: it knows when the
            # bucket refills.
            delay_ms = max(delay_ms, failure.provider_retry_after_ms)

        agent = getattr(failure_payload, "agent", None)
        session = getattr(agent, "session", None)
        if session is not None:
            session.append(
                "llm/retry",
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
