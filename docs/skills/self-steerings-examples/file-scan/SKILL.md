---
name: file-scan
version: 1.0.0
description: Create the workspace's notes files, then scan the tree for TODO/FIXME markers and report what is outstanding.
argument-hint: "[pattern] [max-results]"
allowed-tools: [bash, read, write, grep, write_todos]
parameters:
  pattern:
    type: string
    default: "TODO|FIXME|HACK|XXX"
    hint: Extended-regex alternation of the markers to scan for.
  max-results:
    type: number
    default: 50
    hint: How many matching lines to print before truncating.
steps:
  - "Create NOTES.md and TODO.md at the workspace root if they are absent"
  - "Scan the tree for the marker pattern and report the counts, the files, and the top matches"
---

# File scan

Two files, then a scan. The steps above are the procedure; this is how to carry
each one out.

## Rules

- Print the exact output of each command before summarising it. A summary with
  no output behind it is a claim.
- If a command exits non-zero, report it and stop.
- Never overwrite an existing `NOTES.md` or `TODO.md` — read it first and leave
  it alone if it has content.

## 1. Create the notes files

At the workspace root, create `NOTES.md` and `TODO.md` if they do not exist:

```
# Notes

Add any project notes here.
```

```
# TODO

- [ ] Review the scan results below
- [ ] Address the outstanding markers found in source
```

Report which files you created and which already existed. Do not touch one that
already has content.

## 2. Scan for markers

```bash
grep -rn -E "{{parameters.pattern}}" . \
  --include="*.py" --include="*.md" --include="*.sh" \
  --include="*.yaml" --include="*.yml" --include="*.json" \
  --exclude-dir=".git" --exclude-dir=".venv" --exclude-dir="node_modules"
```

Then report three things: how many matches there were, how many distinct files
they came from, and the first {{parameters.max-results}} matching lines verbatim.
If there are none, say so plainly — an empty scan is a result.
