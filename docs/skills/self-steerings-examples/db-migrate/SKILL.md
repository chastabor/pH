---
name: db-migrate
version: 1.0.0
description: Run database migrations safely across environments — validate, dry-run, review the schema diff, apply to staging, smoke-test, then apply to production with row-count verification.
argument-hint: "<target> [migration-path] [allow-destructive] [dry-run]"
allowed-tools: [bash, read, glob, grep, write_todos]
parameters:
  target:
    type: string
    required: true
    enum: [dev, staging, prod, all]
    hint: Target environment to migrate.
  migration-path:
    type: string
    default: "./migrations"
    hint: Directory holding the migration files.
  allow-destructive:
    type: boolean
    default: false
    hint: "Permit DROP TABLE, DELETE, TRUNCATE — requires explicit opt-in."
  dry-run:
    type: boolean
    default: false
    hint: Validate and preview without applying anything.
  rollback-on-failure:
    type: boolean
    default: true
    hint: Run the down migration immediately if an apply fails part way.
steps:
  - "Validate every pending migration's syntax and ordering, and capture row counts"
  - "Dry-run against dev and report exactly what would change"
  - "Present the schema diff for review — stop here"
  - "Apply to staging after confirmation, and report the result"
  - "Smoke-test the application against the migrated staging database"
  - "Apply to production — ask for approval first, naming what will change"
  - "Verify row counts before and after, and report any discrepancy"
---

# Database migrations

The steps above are the procedure; this is how to carry each one out.

## Rules

- **Never apply to prod without staging having been applied and verified first.**
  If `{{parameters.target}}` is `all`, that means staging completes and smoke-tests
  clean before prod is touched at all.
- **Never run `DROP TABLE`, `DELETE`, or `TRUNCATE`** unless
  `{{parameters.allow-destructive}}` is true. If the dry-run reveals one and it is
  not, stop and report which migration and which statement.
- Never skip the smoke test. A migration that applied cleanly and broke the
  application is still a broken migration.
- Always capture row counts before and after, per table.
- If an apply fails part way and `{{parameters.rollback-on-failure}}` is true, run
  the down migration immediately, then report both failures if the rollback also
  fails.
- When `{{parameters.dry-run}}` is true, nothing is applied anywhere.

## 1. Validate

List the pending migrations under `{{parameters.migration-path}}` in order. For
each: confirm it parses, confirm it has a matching down migration, and confirm
the ordering has no gap or duplicate. Capture the current row count for every
table a migration touches — that is the baseline step 7 compares against.

Report the list, in the order it will apply, and any file that failed validation.

## 2. Dry-run against dev

Apply against dev in a transaction that is rolled back, or with the tool's own
dry-run mode. Report every statement that would run and every object it would
touch.

**Flag any destructive statement explicitly.** If one is present and
`{{parameters.allow-destructive}}` is false, stop here.

## 3. Schema diff — then stop for review

Present the before/after schema difference: tables added, columns added, changed
or dropped, indexes, constraints. Say plainly which changes are irreversible.

Stop. Nothing is applied until the person has read this diff and approved it.

## 4. Apply to staging

Ask for confirmation, naming the environment and the migration list, then apply.
Report each migration's result and elapsed time. If one fails, follow the
rollback rule above and stop.

## 5. Smoke-test

Run the application's smoke tests against the migrated staging database.
Report what passed and what failed. A failure here stops the procedure — prod is
not touched.

## 6. Apply to production

State exactly what will happen: which migrations, which tables, which of them are
irreversible, and the current row counts. Then ask for approval.

Apply only after it is given. Report each migration's result.

## 7. Verify row counts

Re-count every table from step 1 and compare against the baseline. Report the
before, the after, and the delta per table, and call out anything unexpected — a
table that lost rows during a migration nobody described as destructive is the
finding this step exists for.
