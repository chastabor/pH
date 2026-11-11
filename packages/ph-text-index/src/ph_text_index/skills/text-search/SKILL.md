---
name: text-search
version: 1.0.0
description: Find the passage in a document corpus that answers a question, by meaning rather than by wording, and get back the text plus the exact lines it came from.
argument-hint: "<the question you want a passage for>"
allowed-tools: [text_index, text_search, read, grep, glob]
---

# Searching prose you have not read

You have `text_index` and `text_search`. Together they answer questions about a
body of documents — specs, design notes, runbooks, ADRs — where the words you
would `grep` for are not the words the author used.

## Index the corpus once

```
text_index(paths=["docs"])
```

With no `paths` it indexes what this deployment configured, which is the whole
workspace unless a profile pointed it elsewhere. It takes `**/*.md` from a
directory unless you pass `glob`, and `exclude` leaves files out
(`exclude=["**/drafts/**"]`). Re-indexing a
document replaces its passages, so running it again after an edit is correct and
cheap. `forget: true` removes documents instead.

A large corpus is indexed **in the background**: `text_index` then answers
`state: indexing` with how far it has got, and you are told when it finishes.
Do not wait or poll — `text_search` works meanwhile, over what is indexed so
far, and says "Not finished indexing" when that is the case. Search again after
the note if the first answer was thin.

Passages are cut to what the embedding model reads, so the result may say how
many tokens they were held to, or name documents that needed smaller ones. That
is the index working; nothing of a document is dropped.

A file too large to index is **skipped and reported** in `skipped`, not an
error — read that list, because a thin corpus and a corpus that quietly refused
half its files look the same in the results.

## Then ask in your own words

```
text_search(query="how does a deployment choose where an agent writes?", k=5)
```

The whole point is that your phrasing need not match the document's. Ask the
question you actually have. `paths=["docs/seams"]` narrows the search to a
subtree, which is worth doing when you know roughly where the answer lives — it
is faster and it removes near-misses from elsewhere.

Ask a question, not a page. One longer than the embedding model reads is
searched by its beginning only, and the result says how much of it was.

## Read the answer, then read around it

Each hit carries the passage **and** `path:start_line-end_line`. Often the
passage is the whole answer and you are done. When it is not, `read` the file
around that span — the passage is a pointer with its context attached, not a
summary.

`score` is an approximate inner product from a quantized index. Use it to
compare hits *within one result*, never as a confidence: a top hit at 0.31 in a
corpus of unrelated documents may still be the best there is, and a 0.55 may
still be wrong.

## Three failure shapes worth recognizing

**An empty result may mean nothing was indexed.** The result says how many
passages it searched — if that is zero, run `text_index` first.

**Retrieval is approximate.** If the top hits are all near-misses, this corpus
may not contain the answer. Say so rather than choosing the least-wrong passage
and presenting it as the answer.

**It finds passages, not facts.** A passage can be out of date, or an argument
the document goes on to reject. Read enough of the surrounding file to know
which before you rely on it.

## When to use `grep` instead

When you know the literal string. `grep` is exact and complete; this is fuzzy
and ranked. Use `grep` for an identifier or an error message, this for an idea.
