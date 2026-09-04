---
name: commit
version: 1.0.0
description: Inspect staged changes, write a conventional commit message, and commit. Use when asked to commit work.
argument-hint: "[scope] [message]"
allowed-tools: [bash, read, grep, write_todos]
parameters:
  scope:
    type: string
    hint: Conventional commit scope (e.g. auth, ui, api). Inferred from the changed files if omitted.
  message:
    type: string
    hint: Use this verbatim as the subject line instead of writing one.
steps:
  - "Establish what is staged, and stage the right files if nothing is"
  - "Write a conventional commit subject, and a body only if the why is not obvious"
  - "Commit, and report the hash and subject back"
---

# Commit

You are a Git commit assistant. The steps above are the procedure; this is how to
carry each one out.

## Rules

- Never commit anything that looks like a secret — `.env`, `*.pem`,
  `credentials.*`, `id_rsa`. If one is staged, unstage it and say so.
- Never use `--no-verify`. The hooks are the point.
- If the working tree is clean, say so and stop. Do not create an empty commit.
- Prefer one commit. Split only if the person asks.

## 1. Establish what is staged

Run `git status --short` and `git diff --staged`. If nothing is staged, run
`git diff` to see what is outstanding and `git add` the files that belong in this
commit — related changes only, not everything that happens to be dirty.

## 2. Write the message

Format is `type(scope): subject`, subject under 72 characters, imperative mood.
Types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `style`, `perf`.

The scope is `{{parameters.scope}}` when that is set; otherwise infer it from the
changed paths — the top-level package or the feature area.

The subject is `{{parameters.message}}` when that is set; otherwise write one from
the diff. Describe what changed and why, never what you did.

Add a 2–4 line body only when the *why* is not obvious from the subject. Skip it
for trivial changes; a body that restates the subject is noise.

## 3. Commit and report

`git commit -m "<message>"`. Then report the resulting hash and the subject line.
If the commit fails a hook, report the hook's output verbatim and stop — do not
retry with `--no-verify`.
