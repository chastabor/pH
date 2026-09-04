---
name: incident-response
version: 1.0.0
description: Work an incident from first signal to postmortem — gather logs, establish blast radius, confirm scope, mitigate, verify recovery, close, and write it up.
argument-hint: "<service> <severity> [environment] [description]"
allowed-tools: [bash, read, write, glob, grep, write_todos]
parameters:
  service:
    type: string
    required: true
    hint: The service that is failing.
  severity:
    type: string
    required: true
    enum: [P0, P1, P2, P3]
    hint: Incident severity. P0 makes every gate mandatory.
  environment:
    type: string
    default: "prod"
    hint: Which environment the incident is in.
  description:
    type: string
    default: ""
    hint: One line on what was observed.
steps:
  - "Gather logs, metrics and recent deploys for the affected service"
  - "Establish the blast radius — who and what is affected, and since when"
  - "Present the scope for confirmation before anything is changed — stop here"
  - "Apply the mitigation after confirmation, recording exactly what was done"
  - "Verify recovery against the same signals that showed the failure"
  - "Confirm the timeline and declare the incident closed"
  - "Write the postmortem to a file"
---

# Incident response

The steps above are the procedure; this is how to carry each one out. Record the
exact UTC timestamp at the start of every step — the timeline is the postmortem's
spine and cannot be reconstructed afterwards.

## Rules

- **Change nothing before step 3 has been confirmed.** No restart, no rollback, no
  infrastructure change. Diagnosis first: a mitigation applied before the blast
  radius is known is a second incident.
- Never skip verification. An incident is not resolved because a fix was applied.
- Never put a secret, API key, or password in the postmortem.
- When `{{parameters.severity}}` is `P0`, every gate is mandatory. Do not
  auto-proceed through any of them for any reason.
- The postmortem is written to a **file**, not only printed.

## 1. Gather

For `{{parameters.service}}` in `{{parameters.environment}}`: recent logs with the
errors isolated, the relevant metrics either side of onset, and every deploy or
config change in the preceding 24 hours. Report the first timestamp at which the
signal appears — that is the onset, and everything else is measured from it.

## 2. Blast radius

Establish and report: which users or tenants are affected, which downstream
services are degraded, whether data is at risk, and whether the failure is
ongoing or intermittent. Say what you could **not** determine — an unknown named
is a risk managed, an unknown omitted is a surprise later.

## 3. Confirm the scope — stop here

Present the incident as understood: onset, symptom, blast radius, most likely
cause, and the mitigation you propose with what it will do and what it risks.

Stop. This is the gate that stands between diagnosis and action.

## 4. Mitigate

Ask for confirmation naming the exact action, then apply it. Record what you ran,
when, and what it returned. If the mitigation does not hold, say so immediately
rather than escalating on your own.

## 5. Verify recovery

Check **the same signals from step 1** — not a proxy for them. Report the error
rate, the metrics, and a functional check that the service is actually serving.
Give it long enough to be meaningful; a green graph five seconds after a restart
is not recovery.

## 6. Close

Confirm this summary, then declare the incident closed:

- Service: `{{parameters.service}}` · Severity: `{{parameters.severity}}` ·
  Environment: `{{parameters.environment}}`
- What was observed: `{{parameters.description}}`
- What was done, and what confirmed recovery

Summarise the timeline in three to five bullets, each with its UTC timestamp.

## 7. Postmortem

Write `postmortem-<service>-<date>.md` containing: summary, impact with numbers
and duration, the timeline from the timestamps you recorded, root cause,
resolution, what went well, what did not, and the follow-up actions with an owner
each.

Blameless. The document explains the system that allowed it, not the person who
touched it last.
