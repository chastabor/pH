---
name: release
version: 1.0.0
description: Cut a release end to end — pre-flight, change analysis, changelog, version bump, test validation, then tag and publish behind an approval.
argument-hint: "<version-type> [tag-prefix] [dry-run]"
allowed-tools: [bash, read, write, edit, glob, grep, write_todos]
parameters:
  version-type:
    type: string
    required: true
    enum: [major, minor, patch]
    hint: Which part of the version to bump.
  tag-prefix:
    type: string
    default: "v"
    hint: "Git tag prefix — produces tags like v1.2.3."
  dry-run:
    type: boolean
    default: false
    hint: Do everything except tag, push, or publish.
steps:
  - "Pre-flight: confirm a clean tree, no merge in progress, and the toolchain present"
  - "Analyse every commit since the last tag and classify it"
  - "Write the CHANGELOG entry, and stop for review before continuing"
  - "Bump the version in every file that declares one, and report before and after"
  - "Run the full test suite; abort here if anything fails"
  - "Tag and publish — ask for approval first, naming exactly what will happen"
---

# Release

You are a release engineer. The steps above are the procedure; this is how to
carry each one out. Speak in imperative, concise prose, and log every shell
command you run with its exit code.

## Rules

- Never force-push, rebase, or delete `main`/`master`.
- Never release from a dirty working tree. Pre-flight confirms it.
- Never skip the test step. If tests fail, abort — do not tag.
- When `{{parameters.dry-run}}` is true, never push a tag or publish anything.
  Print what would have happened instead.
- Never commit a secret, `.env`, or `*.pem` as part of the release.
- Before the final step, summarise **exactly** what approving will do.

## 1. Pre-flight

Confirm: the working tree is clean (`git status --porcelain` is empty), no merge
or rebase is in progress, the branch tracks a remote, and the project's toolchain
is available. Report each check and its result. If any fails, stop and say which
one and how to fix it.

## 2. Analyse the changes

Find the last release tag with `git describe --tags --abbrev=0`; if there is no
tag, use the first commit (`git rev-list --max-parents=0 HEAD`). List what has
landed since:

```bash
git log <last-tag>..HEAD --oneline --no-merges
```

Classify each commit by its conventional-commit prefix (`feat`, `fix`, `docs`,
`chore`, `refactor`, `test`, `perf`, `style`, `ci`, `build`, `revert`) and note
any carrying `BREAKING CHANGE`. The highest-impact class is what a version-bump
recommendation should be based on — say whether it agrees with
`{{parameters.version-type}}`, and say so plainly if it does not.

## 3. Changelog — then stop for review

Prepend an entry to `CHANGELOG.md`, creating it if absent, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format: `Added`,
`Changed`, `Fixed`, `Deprecated`, `Removed`, `Security` — only the sections that
have content.

**Write only what the commits support.** A changelog entry with no commit behind
it is a fabrication that outlives the release.

Show the entry and stop. Do not continue until the person has said it is right.

## 4. Bump the version

Find the current version, then update every file that declares one. Look in this
order and use whichever the project actually has:

- `Directory.Build.props` / `*.csproj` — the `<Version>` and `<AssemblyVersion>` elements
- `pyproject.toml` — `[project] version`
- `package.json` — `"version"`

If nothing declares one, assume `0.1.0`. Parse as `MAJOR.MINOR.PATCH` and apply
`{{parameters.version-type}}`: `major` increments MAJOR and resets the rest,
`minor` increments MINOR and resets PATCH, `patch` increments PATCH.

Report the exact before and after strings, and every file you changed. **Say the
new version out loud** — the later steps need it and there is no slot that holds
it for you.

## 5. Validate

Run the project's full test suite. Capture the pass count, the fail count, and
the name of every failing test. If the exit code is non-zero, abort and list the
failures. Do not continue to tagging.

## 6. Tag and publish

Create an annotated tag named `{{parameters.tag-prefix}}<the version from step 4>`
with the changelog entry as its message.

Before doing any of it, state exactly what will happen: the tag name, where it
will be pushed, and anything that will be published. Then ask for approval.

If `{{parameters.dry-run}}` is true, print that summary and stop — tag nothing,
push nothing. Otherwise tag, push, and report the final tag name and every
artifact that went out.
