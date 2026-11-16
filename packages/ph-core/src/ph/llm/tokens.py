"""How many tokens text is, for a caller with no tokenizer to ask.

The careful estimate: what sizes an input for a server that refuses one too
long — text-index's passages for an embeddings endpoint, the state a decision
model is asked about. Careful means seldom under, so it counts UTF-8 bytes
rather than characters: a character three bytes long is about a token, and an
estimate in characters sends three times what fits in such a script.

**Not the meter's estimate.** `ph.seams.token_meter.CHARS_PER_TOKEN` answers
another question — about how full a context is, for a threshold that only asks
for a compaction sooner or later — and is a typical case in characters, as its
upstream's is. Neither should take the other's number.

@module ph.llm.tokens
"""

from __future__ import annotations

__all__ = ["BYTES_PER_TOKEN", "bytes_of", "tokens_of"]

BYTES_PER_TOKEN = 3
"""How many bytes of text one token is taken to hold, on the careful side.
English prose runs near four characters a token and code nearer three; a script
that is three bytes a character is about one token a character. So bytes over
three is seldom under."""


def tokens_of(text: str, bytes_per_token: float = BYTES_PER_TOKEN) -> float:
    """About how many tokens `text` is: its UTF-8 bytes over `bytes_per_token`,
    unrounded, so the weights of a text's parts add up to the whole's.
    `BYTES_PER_TOKEN` unless the caller has learned how dense its text runs."""
    return len(text.encode("utf-8")) / bytes_per_token


def bytes_of(tokens: int) -> int:
    """How many UTF-8 bytes of text `tokens` tokens are taken to hold —
    `tokens_of` the other way round, for a caller sizing an input to a limit."""
    return tokens * BYTES_PER_TOKEN
