---
name: code-graph
version: 1.0.0
description: Understand an unfamiliar codebase by asking its graph — what a name is, who calls it, what a change breaks, what is biggest — instead of reading files to find out.
argument-hint: "<what you are trying to find out>"
allowed-tools: [code_index, code_graph, read, grep, glob]
---

# Reading a codebase by asking it

You have `code_index` and `code_graph`. They answer questions about structure
and they **never return source code** — every answer is a `path:start-end` you
then `read`. Used in that order they replace most of the reading you would
otherwise do to find out which two files mattered.

## Index first, once

```
code_index(paths=["packages/thing/src"])
```

Point it at a package, not a monorepo. It re-parses only files whose contents
changed, so running it again after your own edits is cheap and keeps every
later answer current — do that rather than reasoning about a stale graph.

If a query says no index exists, this is the step you skipped.

## Then ask the question you actually have

| you want to know | call |
|---|---|
| "there is something about X here" | `code_graph(mode="search", query="X in your own words")` |
| "where exactly is `foo`" | `code_graph(mode="define", query="foo")` |
| "what calls `foo`" | `code_graph(mode="callers", query="foo")` |
| "what does `foo` use" | `code_graph(mode="callees", query="foo")` |
| "what breaks if I change `foo`" | `code_graph(mode="impact", query="foo", distance=2)` |
| "what are the big pieces here" | `code_graph(mode="entities", path="pkg/", kind="class")` |

`search` takes **prose** and matches names and docstrings, so ask it the way you
would ask a colleague. `define`/`callers`/`callees`/`impact` take an **exact
symbol name**.

## Then read the narrow thing

Every row carries `path`, `start_line`, `end_line`. Feed those to `read` — with
`offset`/`limit` around the span — rather than opening whole files. For
`callers` and `callees`, `ref_path:ref_line` is the *call site* and
`path:start_line` is the *definition*; they are usually different files, and the
call site is normally what you want to see first.

## Two things to hold in mind

**Matching is by name.** Two `register` methods on two classes are one name to
this index. When a result reports `definitions` greater than 1 the answer may
mix them — run `mode="define"` to see the candidates and pick, rather than
assuming the first is yours.

**It is structure, not behaviour.** It cannot tell you what a function does,
whether a branch is reachable, or what a value is at runtime. It tells you where
to look. When the answer you need is *why*, read the code and its comments.

## When to use `grep` instead

When you know the exact string — an error message, a config key, a literal. This
tool is for when you know the *idea* and not the spelling. They compose well:
`code_graph` to find the region, `grep` to find every occurrence inside it.
