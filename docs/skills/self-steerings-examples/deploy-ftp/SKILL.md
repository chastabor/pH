---
name: deploy-ftp
version: 1.0.0
description: Build the project, diff the output against a remote FTP server, and upload only what changed — pausing for review before any transfer.
argument-hint: "<host> <user> [remote-path] [local-path] [dry-run]"
allowed-tools: [bash, read, glob, grep, write_todos]
parameters:
  host:
    type: string
    required: true
    hint: FTP hostname or IP address.
  user:
    type: string
    required: true
    hint: FTP username. The password comes from the environment and is never printed.
  remote-path:
    type: string
    default: "/public_html"
    hint: Absolute path on the remote server to deploy into.
  local-path:
    type: string
    default: "./dist"
    hint: Local directory to upload. Must exist after the build step.
  build-command:
    type: string
    default: "npm run build"
    hint: "Command that produces the build output. Set to 'none' to skip the build."
  dry-run:
    type: boolean
    default: true
    hint: Report what would transfer without transferring anything.
steps:
  - "Pre-flight: confirm the credentials, the remote, and the local paths are usable"
  - "Build the output, unless the build was explicitly skipped"
  - "Diff local against remote and present the manifest for review — stop here"
  - "Upload the manifest after approval, and report what actually transferred"
---

# Deploy over FTP

The steps above are the procedure; this is how to carry each one out.

## Rules

- **Never upload** `.env`, `.env.*`, `*.pem`, `*.key`, `*.p12`, or
  `credentials.*`. Filter them out of the manifest and say that you did.
- **Never read or print `FTP_PASSWORD`.** Pass it through the environment. It must
  not appear in a command line, a log line, or your own output — the session log
  keeps whatever it is told.
- Never delete a remote file unless the person explicitly asked for it.
- Never transfer anything without an explicit approval at the diff step.
- Use passive mode unless told otherwise.
- When `{{parameters.dry-run}}` is true, report and transfer nothing.

## 1. Pre-flight

Confirm, and report each one: `FTP_PASSWORD` is set (that it exists — never its
value), the host `{{parameters.host}}` resolves, an FTP client is installed, and
`{{parameters.local-path}}`'s parent is writable. If a check fails, stop and say
which and how to fix it.

## 2. Build

If `{{parameters.build-command}}` is `none`, skip this and say so. Otherwise run
it, and confirm `{{parameters.local-path}}` exists and is non-empty afterwards. A
build that "succeeded" with no output is a failed build.

## 3. Diff — then stop for review

List the remote tree under `{{parameters.remote-path}}` and compare it to
`{{parameters.local-path}}` by size and modification time. Produce a manifest in
three parts: files to add, files to update, files that are identical and will be
skipped.

Remove any file matching the secret patterns above and say what you removed.

Present the manifest with a total file count and total bytes, and stop. This is
the gate — nothing transfers until the person approves this list.

## 4. Upload

Upload exactly the manifest from step 3 — nothing that was not on it. Pass the
host, user, remote path and local path through the environment; the password is
already there.

Report each file transferred, the total count, total bytes, and elapsed time. If
`{{parameters.dry-run}}` is true, print that summary and transfer nothing.
