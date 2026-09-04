---
name: pr-ready
version: 1.0.0
description: Get a branch ready to open a pull request — sync check, tests, lint, commit what is outstanding, write the description, then open it.
argument-hint: "<target-branch> [reviewers] [labels] [draft]"
allowed-tools: [bash, read, write, edit, glob, grep, write_todos]
parameters:
  target-branch:
    type: string
    default: "main"
    hint: The branch this PR will merge into.
  reviewers:
    type: string
    default: ""
    hint: Comma-separated reviewers to request.
  labels:
    type: string
    default: ""
    hint: Comma-separated labels to apply.
  draft:
    type: boolean
    default: false
    hint: Open the PR as a draft.
steps:
  - "Sync check: confirm this branch can legitimately open a PR against the target"
  - "Run the full test suite and stop if anything fails"
  - "Run the linter and formatter, and fix what they flag"
  - "Commit anything still outstanding, or confirm the tree is already clean"
  - "Write the PR description from the actual commits, and stop for review"
  - "Open the PR after approval and report its URL"
---

# PR ready

The steps above are the procedure; this is how to carry each one out.

## Rules

- Never open a PR while tests are failing. Step 2 is a hard stop.
- Never open a PR from `main`, `master`, or `{{parameters.target-branch}}` itself.
- Never include a secret, a `.env`, or a credential in the branch or the PR body.
- **Base the description on the commits that exist.** Do not describe a feature
  nobody wrote — a reviewer reads the description and trusts it.
- If the branch has no commits ahead of the target, stop and say so.
- Never force-push during this procedure.

## 1. Sync check

```bash
git symbolic-ref --short HEAD
git fetch origin {{parameters.target-branch}} --quiet
git rev-list --left-right --count origin/{{parameters.target-branch}}...HEAD
```

Refuse to continue if the current branch *is* `{{parameters.target-branch}}`, or
`main`, or `master` — say a feature branch is needed. Report how far ahead and
behind the branch is. If it is behind, say so and let the person decide whether to
rebase; do not rebase for them.

If the branch is zero commits ahead, stop. There is nothing to open a PR for.

## 2. Tests

Detect the project's test runner and run the whole suite. Report the pass count,
the fail count, and the name of every failure. **Hard stop on any failure** — do
not continue to lint, and do not open a PR.

## 3. Lint

Run the project's linter and formatter. Fix what they flag, then re-run to
confirm clean. Report anything you could not fix and why — a lint failure you
worked around is worth a sentence.

## 4. Commit what is outstanding

```bash
git status --short
```

If there are uncommitted changes, commit them now — a conventional
`type(scope): subject`, secrets excluded, no `--no-verify`. If the tree is
already clean, say "working tree clean — nothing to commit" and move on.

## 5. Write the description — then stop

Read the actual commits (`git log origin/{{parameters.target-branch}}..HEAD`) and
write:

- a one-paragraph summary of what this branch does and why
- a bullet per meaningful change, grouped if there are many
- how it was verified — the real test and lint results from steps 2 and 3
- anything a reviewer should look at first

Show it and stop. Do not open the PR until the person has approved the text.

## 6. Open it

Open the PR against `{{parameters.target-branch}}`, requesting
`{{parameters.reviewers}}` and applying `{{parameters.labels}}` when those are
set, as a draft when `{{parameters.draft}}` is true. Report the URL.
