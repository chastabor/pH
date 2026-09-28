# Session profiles — a starting profile, logged overrides, and a rebuild on restart

*2026-09-26. Follows L7 in `plans/Two_Log_Facts_Todo.md` ("`/sandbox allow` lives in a
profile drop-in, not the session log"), which widened, on review, from an audit of sandbox
allowances into an audit of the whole working environment. The decisions below are the
user's, taken after the options review of 2026-09-25.*

**Status, 2026-09-27.** S1–S7b are done: the `affects` declaration, the layers by kind
(decisions 23 and 24), the `models` row, the named profile format with `phern profiles
show|fold`, each root's `profile/base` in its log, overrides through one door, each
root mounted from its own log (S5), a changed named profile held, listed and adopted
on purpose (S6), `/profile show|diff|save|use|clear` (S7), a child's model from its
parent's list and a profile its parent assigns (S7b), and reading the audit (S8).
Every step is done; what each left open is under its own entry, and the follow-ups
after S8 closed most of it (2026-09-27). Still open: a child holding fewer hosts than
its parent, and the named-profile store as a deployment row.

## The goal

A person tunes a profile in a TUI session, saves it as a named profile, and a headless
daemon runs autonomous work on it. Every session can say, at any point in its log, which
environment the agent worked in and how it got there, and a restarted session rebuilds
that environment rather than whatever the daemon happens to have loaded.

"Environment" means what the agent can see or do:
- the model and its route;
- the skills it can read;
- the tools it is shown;
- the sandbox (mode, allowances, network, containment tier);
- the workspace setup;
- the approval and permission posture.

## Decided with the user

1. **Profiles are per session**, and they are the main tool a person uses to manage one.
   A profile is fine-tuned in a TUI session and saved to be run headless.
2. **Profiles are sparse.** Every row has default settings, so a profile file holds only
   what differs, not every row.
3. **Only deviations are logged, as overrides.** Overrides come from command-line start
   options or from commands (`/sandbox` and the others that change the loaded settings). A
   change that matches the current setting records nothing.
4. **Base and overrides are kept apart.** That separation lets the daemon detect when the
   base profile has changed.
   - The base is saved with the session, in full (decision 8).
   - The session rebuilds from it on startup.
5. **Save as.** A useful profile can be saved as a new named profile in `$PH_HOME/profiles/`.
   - Saving merges the base with the overrides.
   - If the new profile is used as the base, it is logged as the base from that point on.
   - The person can keep applying overrides or clear them.
   - Swapping base profiles keeps the overrides.
6. **Only the environment belongs in the session profile.** Settings that don't affect the
   agent's working environment, such as TUI configuration, leave the session profile and move
   to the TUI's own configuration.
7. **A restart whose named profile has changed holds and asks.** This is when the file in
   `$PH_HOME/profiles/` differs from the copy saved with the session.
   - **No:** the session starts unchanged, on its saved base and its overrides.
   - **Yes:** the new version is saved with the session and loaded, and then the logged
     overrides and the command-line options are applied.

Answered on 2026-09-26, the first draft's open questions:

8. **The named profile is sparse; the profile saved with the session is full.** A named
   profile acts as the defaults for its sessions. What is saved with a session is the full
   state that gets loaded: every row and every setting, defaults included. Comparing it with
   the named profile as it composes now both detects a change and shows what changed. (How
   defaults are generated was examined for this; see below.)
9. **Command-line options do not affect the hold and ask.** They are applied after the base
   profile is checked and the question answered, whichever way it went.
10. **A daemon keeps its sessions' profiles by default.** Updating them has to be
    intentional. It is an operation of its own, runnable without starting a session, that
    moves every session based on a named profile to its new version. Each session keeps its
    overrides, which are applied to the updated profile.
11. **Overrides are saved only by a profile save.** Using a different named profile
    replaces the base that gets loaded, and the overrides are then applied to it or cleared,
    as the person chooses at that time.
12. **Skill versions are for the audit only, logged at runtime.** A skill's version never
    stops a task from running or makes a restart ask, since skills are meant to keep being
    refined.

From the review of 2026-09-26:

13. **A change is listed, not only detected.** Before accepting a changed profile, a person
    sees what changed: each setting with its old and new value, so the decision is an
    informed one.
14. **A pH version change is reported, not asked about.** A new version of the code is a
    bigger change than a configuration one, and there is little a person can do about it.
    The version is recorded and shown, and the settings its new defaults would change are
    listed for information.
15. **No hash.** Measured: composing and resolving rlm takes 13 ms, which a restart does
    anyway to mount it, and comparing that with the saved full profile takes 0.13 ms. A
    hash would not make detection faster, and it would be one more thing to keep right.
16. **A full listing of a profile** is for debugging, and for a first session, deciding
    what to change.

Follow-ups, 2026-09-26:

17. **The saved profile lives in the log, with the log's guarantees.** It is written through
    the one door (`LogWriter`), as a single record or one `Session.batch()`, and must stay
    recoverable after a failure: the torn-tail and take-back rules, and a batch whole or not
    at all, cover it as they cover every record.
18. **A change is found only when a session starts.** The daemon does not check periodically;
    profile changes are intentional. `adopt` is the intentional answer given ahead of time:
    a running session keeps its base, and its next start applies the adopted version. The
    first draft's question about restarting a live session falls away.
19. **A subagent runs on its parent's profile, or on one the parent assigns.**
20. **A profile lists models, one of them the default.** The parent runs on the default.
    The parent, or a skill, may pick another listed model when starting a subagent, for
    example a skill that needs a model suited to categorization rather than general text.
    The text index should later take its model options from the same list.
21. **A profile a parent assigns is a narrowing.** It is applied on the parent's mount, and
    a child is never wider than its parent. A child does not get a mount of its own.
22. **The existing drop-ins are folded on purpose**, with `phern profiles fold`, as item 0
    describes.
23. **Three kinds of configuration, one owner each** (taken during S1). A row's `affects`
    says which:
    - `environment`: the session profile, logged and audited;
    - `presentation`: the TUI's configuration;
    - `deployment`: the daemon's or host's own configuration. Telemetry lives there, and so
      do the locations the other two are kept in, such as `$PH_HOME/sessions` and
      `$PH_HOME/profiles`.
24. **The files for the three** (taken during S1).
    - Deployment: `$PH_HOME/daemon.yaml`, with a `paths:` block (`sessions`, `profiles`)
      and `rows:` in the profile grammar, deployment rows only. Read once per process.
      The daemon's own flags (`--max-concurrent-children`) set deployment rows after it.
    - Presentation: no layer a person writes. pH's presentation rows ship in every named
      profile; a person hides a screen from one terminal in `tui.json`
      (`hidden_screens`), which the daemon never reads, so a remote TUI works the same.
    - Environment: the session profile, as before; `--patch` sets environment rows only.

## What exists today, and what each decision changes

- **Named profiles** are a table in code (`ph_app/profiles.py:PROFILES`), with two things
  layered over each:
  - `$PH_HOME/profiles/<name>.yaml`, a person's overlay;
  - `$PH_HOME/profiles/<name>.d/*.yaml`, drop-ins pH writes; `/sandbox` is the only writer.

  The shipped profiles (base, headless, tui, rlm, …) stay in code as the layer that supplies
  defaults (decision 2). A person's named profile becomes one file that extends a shipped
  profile and lists only the rows that differ. Drop-ins retire (decisions 3 and 5).
- **Granularity is the row.** A patch replaces a row's whole config, deliberately, so a
  row's effective value is always one layer's and readable in one place (`_apply_patch` in
  `ph/cordis/loader.py`). An override is therefore a row's new config, and "sparse" means
  only the rows that differ. That is consistent with the loader's rule. Most rows already
  set no config and use their plugin's defaults.
- **One profile per daemon.** The daemon composes a single profile at startup
  (`Supervisor.profile`) and mounts it for every root. `session/new` has no profile
  parameter, and no session records its profile: the header holds `cwd` only.
  Decisions 1 and 4 make the profile per session.
  - This also fixes a problem found in the review: `/sandbox`'s "applies now and on the
    next start" was false within one daemon. A root started later, including this one after
    it goes idle and comes back, mounted without the change.
- **The model is not a profile setting.** It comes from the daemon's `--provider` and
  `--model` flags, one pair for every root. The TUI's `/model` changes only the TUI's label.
  The model needs a row (provider, model, reasoning effort) so it is part of the profile;
  the flags then become command-line overrides and `/model` a real one.
- **Skills** are whatever `SKILL.md` files the `skills-progressive` row's `paths` hold. The
  profile names the paths; the files can change underneath it.
- **The `tui` profile mixes two kinds of row.**
  - The sandbox default (`workspace-write`) and `tool-ask-user` shape what the agent can do:
    environment.
  - `tui-screen-trajectory` only adds a screen: presentation, which decision 6 moves out.
- **The session directory.** Logs live in `$PH_HOME/sessions/<family>/`, where the family is
  `<cwd-tag>-<root id>` and is shared by the root, its forks and its children. That is where
  a readable copy of the saved profile goes when one is asked for (decision 17). Children and forks
  run on the root's mount, so they use the root's base and overrides; a child's log records
  which base its root was on.
- **The log already records part of the environment.** `request/header` holds the model
  config, the system prompt and the tools the model saw. `request/context` holds the route.
  `permission/preset`, `sandbox/mode` and `approval/policy` record posture changes, and
  `subagent/admitted` records what a child asked for.
  - Not recorded: the profile, sandbox allowances, network, the containment tier, the
    defaults in force before any posture event, and a root's tool and skill restrictions.
  - `/sandbox` leaves only the prose of `command/run` and `command/done`.

## How defaults are generated (examined 2026-09-26, for decision 8)

A row's effective config is built in three tiers:

1. **The plugin's config model.** `PluginSpec.resolve_config` (`ph/cordis/plugin.py`)
   validates the row's raw config against the plugin's model, so every field the profile
   leaves unset takes the model's default. A row with no config at all gets `model()`,
   every field at its default. Most rows are like that: 75 of rlm's 88 set nothing, and 65 of
   headless's 71.
2. **The shipped profile's YAML** (`ph/bundles/*.yaml`, `ph_app/profiles/*.yaml`, a bundle
   such as ph-rlm's), which sets the config of some rows.
3. **The person's layers** over it: today `$PH_HOME/profiles/<name>.yaml`, the drop-ins,
   and `--patch`. The last layer to set a row replaces that row's whole config.

What this means for decision 8:
- **Comparing the person's file alone would miss two of the three tiers.** A pH upgrade can
  change a shipped profile's YAML or a plugin's default, and the named profile would then
  produce different settings under an unchanged file.
- **So a change is found by composing the named profile as it is now,** every row resolved
  through its model, and comparing it with the full profile saved with the session.
  - Measured: every row of headless and rlm resolves, the full state is 6.9 KiB and 9.2 KiB,
    composing rlm takes 13 ms and the comparison 0.13 ms (decision 15).
- **Each changed setting can be attributed.** A row whose config the person's file sets
  changed because they edited it. Any other change came from pH: its shipped YAML or a
  plugin's default, which is decision 14's case.
- A saved state whose fields a newer pH no longer accepts, such as a field renamed in a
  plugin's model, fails to validate on rebuild. It is refused by name, the rows and fields
  listed, rather than loaded partly.

## The design

**0. Folding the existing drop-ins** (decision 22).
- **What they are.** `/sandbox` is the only writer of `$PH_HOME/profiles/<name>.d/`. It
  writes one file, `sandbox.yaml`: a comment header and one row carrying the whole
  `sandbox-allow` config. The loader reads `<name>.yaml`, then each `<name>.d/*.yaml` in name
  order, and the last layer to set a row wins. So in practice a profile has at most one
  drop-in, and it replaces one row.
- **Folding one profile**, with `phern profiles fold [name]`, run on purpose:
  1. Compose the profile as it composes today, drop-ins included, and keep the result.
  2. Append each drop-in's rows to the end of `<name>.yaml`, under a comment saying where they
     came from and when. The file is created, extending the shipped profile, if the person had
     none. Appended as text, so the person's own comments and layout stay. The loader lets a
     later entry for a row replace an earlier one in the same file (checked), so the appended
     row wins exactly as the drop-in did.
  3. Compose again, and compare with step 1. Any difference restores the file and stops,
     naming the row.
  4. Move `<name>.d/` aside to `<name>.d.folded-<date>/`, which nothing reads.
- **Why no session notices.** The composed profile is the same before and after, so the next
  start's comparison finds no change, and nothing asks. A session from before this plan has no
  saved base; its first start records one.
- **Until a profile is folded**, the loader keeps reading its `.d/` directory, and
  `phern agents doctor` names it, so no setting is dropped and none is read silently.
- One thing folding does not change: the drop-in replaced the whole `sandbox-allow` row, so it
  pinned pH's default hosts as they were when it was written. Its header already says so, and
  the folded row pins them the same way.

**1. A named profile file.** `$PH_HOME/profiles/<name>.yaml`:
```yaml
extends: rlm              # a shipped profile, which supplies every default
rows:
  - id: sandbox-allow
    config: {network: {mode: allowlist, hosts: [pypi.org, docs.python.org]}}
  - id: agent             # the new model row
    config: {provider: anthropic, model: claude-sonnet-5}
```
Only rows that differ from the profile it extends. Composing it is the loader's own
semantics: the shipped layers, then this file's rows. `Profile.dump()` must round-trip into
this format, which it does not today (it adds a `layer` key the loader refuses).

**2. The session's base.** When a session is created, the named profile is composed and
every row resolved through its model (decision 8). A required
`profile/base {name, rows, sources, phVersion}` record is logged, where `rows` is that full
state and `sources` names the person's files it was composed from.
- **Saved in the log itself** (decision 17). About 7–9 KiB, durable with the log and ordered
  with the overrides. Forks and children inherit it, and there is no separate file to lose or
  keep in step. `phern profiles session <id>` writes it out as YAML for a person to read.
- **On disk before the agent's first step.** The record is written, then the session is
  written (`session_written`), then the mount acts on it. A crash before that point leaves a
  session with no base, which is one that never ran: its first start composes the named
  profile then.
- **A switch of base is one batch.** `/profile use`, a "yes" on restart and an adopted version
  log `profile/base` together with the `profile/override-cleared` records it implies, in one
  `Session.batch()`, so a crash leaves the old base with its overrides or the new one with
  its own, never a mixture. Rebuilding reads the latest base.

**3. Overrides.** A required `profile/override {row, config | disabled, source, command}`
record, where `source` is `cli`, `command` or `verb`. A `profile/override-cleared {row}`
record removes one.
- Written only when the value differs from the effective setting (decision 3).
- Never written to a named profile except by `/profile save` (decision 11).
- **On disk before it takes effect**, the rule the F findings were about: a `/sandbox allow`
  that reached the agent but not the log would leave a session that ran with a host allowed
  and a log that never says so. So the record is written and the session written before the
  row is reconfigured, and a change whose record cannot be written is not applied, as `!!`
  does with a command it cannot record.
- `Mount.reconfigure` is the one door: it applies the change and writes the record, so no
  command can change a row without the log saying so.
- `/sandbox allow` becomes an override and stops writing a drop-in.
- The existing posture records (`permission/preset`, `sandbox/mode`, `approval/policy`)
  stay as they are. They are already structured, and the audit reads them as overrides.

**4. Rebuild on startup.** The supervisor mounts each root from its own log: the base the
latest `profile/base` holds, then the overrides in log order, then this start's
command-line options (logged as overrides where they differ). `session/new` gains a
`profile` parameter. The daemon's `--profile` becomes the default for a new session, not
the profile of every root.

**5. A changed named profile on restart** (decisions 7, 9, 10, 13, 14 and 18). Only when a
session starts, never while it runs, the named profile is composed as it is now and compared
with the base saved with the session.
- **The difference is listed** (decision 13). Each changed setting is shown with its old
  and new value; for a list, what was added and removed rather than the whole list. Each is
  marked as coming from the person's file or from pH, and the session overrides that still
  apply over a changed row are named:
  ```
  rlm has changed since this session started:
    sandbox-allow  network.hosts      + developer.mozilla.org          (your profile)
    workspace      tier               worktree → overlay               (your profile)
    llm-retry      maxAttempts        3 → 5                            (pH 0.4.0 → 0.5.0)
  Still applied over it, from this session:
    sandbox-allow  network.hosts      + docs.rs
  ```
- **The person's file changed, with a person there to ask** (the root is started for a front
  end that takes asks): hold with that list, `needs-profile-decision`, through the ask desk.
  - **No:** it starts on the saved base.
  - **Yes:** the new version is saved with the session, and `profile/base` is logged.
- **Only pH's defaults changed** (decision 14): no question. The session keeps its saved
  settings, which are explicit values the new version still reads. The list is shown with
  the version change, and taking the new defaults is the same deliberate step as adopting.
- **Unattended** (the daemon starting a root for a schedule or autonomous work): keep the
  saved base, without asking. `phern agents doctor` and `sessions/list` say a newer version
  exists and how many settings differ.
- **Adopting on purpose.**
  - `phern profiles diff <name> [--session <id>]` lists what adopting would change.
  - `phern profiles adopt <name>`, with `--session <id>` or for every session based on
    `<name>`, shows that list and records the answer (`--yes` for a script): an ignorable
    `profile/adopted {name, rows}` in each session, the version that was accepted.
  - The next start applies it: that version becomes the base and `profile/base` is logged
    then, so a base record always marks when a base took effect. Each session keeps its
    overrides, applied to the new base. A running session is not interrupted (decision 18).
  - If the named profile changed again after the adopt, the next start compares against the
    adopted version and treats the difference as any other change.
  - Needs no session started. A session a running daemon holds is updated through the
    daemon; one nothing holds is updated under its lease.
- After the base is settled either way, the logged overrides are applied, then this start's
  command-line options (decision 9).

**6. Commands.**
- `/profile show`: the base, the overrides, and the effective setting of each row.
- `/profile diff`: this session's saved base against its named profile as it composes now,
  as in item 5.
- `phern profiles show <name> [--full]`: what the named file sets, or with `--full` every row
  and every setting resolved, defaults included (decision 16). For debugging, and for
  deciding what to change before a first session.
- `/profile save <name>`: writes the base merged with the overrides as a sparse named
  profile, and logs `profile/saved`.
- `/profile use <name>`: replaces the base with another named profile and logs
  `profile/base`; the person chooses there whether the overrides are applied to it or
  cleared (decision 11).
- `/profile clear [row]`: clears one override or all of them.

After a save that is then used as the base, overrides equal to the new base no longer
deviate and are cleared (decision 3's own rule).

**7. Presentation out of the profile** (decision 6). Each plugin declares, at build time,
whether it shapes the agent's environment or only a front end's presentation (an `affects`
field on its declaration).
- Presentation rows (screen registrants such as `tui-screen-trajectory`, and TUI status
  readings) move to the TUI's configuration.
- The session profile and its overrides cover environment rows only.
- A gate refuses a row that declares neither.

**8. Models, listed in the profile** (decision 20). Today a route is two free-form strings,
`provider` and `model`: the daemon's flags for a root, and `SubagentRequest.provider`/`model`
for a child, else its parent's (`child_route`). The adapter rows configure what each provider
can reach. The profile gains a `models` row:
```yaml
- id: models
  config:
    default: main
    models:
      main:     {provider: anthropic, model: claude-sonnet-5, reasoningEffort: medium}
      fast:     {provider: anthropic, model: claude-haiku-4-5}
      classify: {provider: llama, model: qwen-classifier}
```
- The root runs on `default`. `/model <key>` becomes a real override of it, and
  `--provider`/`--model` a command-line one.
- A spawn names a model by its key, and a skill's front matter may name the one it needs
  (`model: classify`), which a spawn the skill drives uses. A key the profile does not list is
  refused, so the models an agent can reach are the ones its profile says.
- A child's admission records the key and the route it resolved to, for the audit.
- Later, the text index takes its model options from the same list. Its model today is an
  embedding model named in its own row (`text-index-local`), so an entry needs a kind
  (chat, embedding), and the list a way to name the default for each kind.

**9. Subagents' profiles** (decision 19). A child runs inside its parent's scope and on its
parent's mount, so the profile a parent assigns is applied there, as a narrowing: its tools,
skills, model (from the list), sandbox and workspace access, each checked against the
parent's by the ceiling that already refuses a child wider than its parent (`check_grant`).
An assigned profile that would widen anything is refused at the spawn, naming the row. A
child never gets a mount of its own (decision 21).

**10. Skill versions, for the audit** (decision 12). When a skill's body is read at runtime,
an ignorable record names the skill and the hash of the content read. It never blocks a
task and never makes a restart ask; the base records only the skill directories
(`skills-progressive.paths`).

**11. Reading the audit.** "The environment at seq N" is a fold: the base at N plus the
overrides up to N. It is shown by `phern profiles session <id> [--at seq]` and by the
trajectory view.

## Todo list

In order. Same rules as the last lists: every gate sabotage-checked, the four gates green,
nothing committed by Claude.

- [x] **S0 — Settle the open questions.** Done 2026-09-26: decisions 8–22.
- [x] **S1 — Environment and presentation apart.**
  - [x] The `affects` declaration on every plugin, and a gate that none is undeclared.
    Done 2026-09-26. `Affects` is a closed `Literal` of the three kinds (decision 23),
    required by `plugin()` with no default, and refused by name by `normalize_plugin` for
    a duck-typed plugin that lacks it. The 101 shipped rows: 77 environment, 21
    deployment, 3 presentation. `phern config` shows the kind. The gate
    (`test_catalog.py`) also pins the families whose kind follows from their name:
    `tui-*` presentation; `*-invariant`, `session-persistence-*`, `session-telemetry*`
    deployment. Each is sabotage-checked.
  - [x] The layers by kind, and presentation rows out of the profile. Done 2026-09-26, as
    decision 24. `ProfileDocument` is a dataclass with `sets`, the one kind a layer may
    touch; `compose_rows` refuses an entry that adds, patches or removes a row of
    another kind, naming the file it belongs in (`CONFIGURED_IN`). Shipped layers and a
    file profile (`--profile x.yaml`, a whole composition until S2's `extends`) set
    any kind. `ph.host` reads `daemon.yaml`; `PathRoots.sessions_dir()` and
    `profiles_dir()` follow its paths, so the persistence rows' default and the
    daemon's cold browse agree. The trajectory screen moved from `tui.yaml` to
    `profiles/presentation.yaml`. Gates: `test_config_owners.py`,
    `test_host_config.py`, the loader's kind tests and `test_tui_screens.py`'s hidden
    screen, each sabotage-checked.
  - [x] The `models` row (item 8), with `/model` and `--provider`/`--model` as overrides of
    its default. Done 2026-09-26. `ph.seams.models`: `Config` (`default` plus `models`
    by key, each a `ModelRoute` with its call settings), `ModelChoice` (a key, a whole
    route, or neither), and one rule, `ModelList.resolve`, that the command line,
    `/model` and rpc all ask; `choose(ctx, …)` adds the served-provider check only a
    mount can make. `ph-base` lists nothing (it has no adapter); `headless` lists
    `main: fake/fake-1`; each provider profile lists its own. `AgentDriver.reroute`
    moves a running agent from its next request, logging a `request/header` change.
    Over the daemon, `session/model` (a mutation, resolved before its key is claimed)
    and `models/list`; protocol 5. The TUI's picker lists the profile's keys and is
    real, and its own `--provider`/`--model` ride `session/new`, so the daemon it spawns
    carries no route. The key travels with the route in `AgentOptions.model_key`, and
    `ModelEntry` is the one shape for both on the wire. `phern daemon` refuses an unlisted `--model`
    before binding, and now takes `--patch`, which the spawned argv already passed.
    Two points settled in the building:
    - A person's choice is not bounded by the list: a whole route runs as given, with
      no key. The list bounds what an agent picks (S7b).
    - The choice does not yet survive a restart: a resumed root starts on its default.
      That is S4's override record.
  - *Gate:* every shipped row declares; a presentation row in a session profile is refused.
- [x] **S2 — The named profile format.** Done 2026-09-26.
  - `extends` plus sparse rows; `Profile.dump()` round-trips.
  - `phern profiles show <name> [--full]`.
  - `phern profiles fold`, as item 0; the loader reads a `.d/` directory only until its
    profile is folded, and the doctor names one that is not.
  - *Gate:* compose, save, reload and compose again give the same rows; a saved profile
    holds only rows that differ.
  - Built: `ph_app.named_profiles` (the format), `sparse_entries` and `Row.to_entry` in
    the loader (the round trip: a dump keeps its `layer`, an entry is what a document
    declares), `Profile.resolved(kinds)` (every row through its model — what `--full`
    prints and S3's `profile/base` will hold), `save_named_profile`, and `phern profiles
    show|fold`. Settled in the building:
    - `extends` names a shipped profile only, one level, and defaults to the file's own
      name when that is a shipped profile; a person-only name is offered by
      `available_profiles` like a shipped one.
    - A `--profile ./x.yaml` in this format is a named profile living elsewhere; a list
      there is still a whole composition, which is what scenario files are.
    - The list format before S2 is read until folded, like the drop-ins, and the doctor
      names both; fold converts it by indenting it under `rows:`, comments kept.
    - Fold uses the one composing door before and after (the drop-ins moved aside
      first), puts everything back on any difference, and carries each drop-in's header
      — `/sandbox`'s pinned-hosts warning — into the folded block.
    - `show --full` lists the session's environment rows; the host's are `phern config`'s.
    - Not yet: `/sandbox` still writes a drop-in until S4 makes it an override, so a
      folded profile can gain one again, and the doctor names it again.
- [x] **S3 — The session's base, recorded.** Done 2026-09-26.
  - The resolved composition (every row through its model) in `profile/base`.
  - No behavior change yet: the audit only.
  - *Gate:* a new root's log holds its full profile, and mounting from that alone gives the
    same rows; a changed plugin default under an unchanged person's file is found by the
    comparison and attributed to pH.
  - Built: `ph.session_profile` — `base_of` (the environment rows through their models,
    the person's source layers, `ph.__version__`), `record_base` (through its own
    `_LOG`, then `session_written`, before the agent's first step), `saved_base`,
    `differences` (setting by setting, dotted paths, `by: person | pH`) and `rebuilt`
    (the base's rows over a host's rows of every other kind). `profile/base` is in the
    known vocabulary as a required type. `open_session` records it, the one door every
    root opens through. `phern profiles session <id>` reads it back from the file on
    disk; the plan's `phern sessions profile` is spelled under `profiles`, since there
    is no `sessions` group. Settled in the building:
    - The base is the named profile *without* its command-line start options:
      `ProfileDocument.override` marks a `--patch`, which S4 logs as an override.
    - Roots only: a child (`parent_session` set) runs on its root's mount, and a fork
      inherits its root's base with the prefix it continues.
    - Once: a session resumed with a base keeps it (S6 decides a changed profile); one
      from before this record gets its first on its next start.
    - Attribution compares the person's own layers as recorded then and as they are
      now, per row, so pH moving a default beneath an unedited file is pH's even when
      the file sets that row.
    - A full row states every field, `disabled` and `config` included; a disabled row
      whose config its model refuses keeps the config as written, since it never mounts.
    - `--mode json` streams what the open itself committed before the stream attached —
      a new log's base — so the stream is still the log from `seq` 0.
    - Values are recorded as they run, `${env:...}` interpolated; a row's config names
      a credential and never holds one, so the record carries no secret.
- [x] **S4 — Overrides, through one door.** Done 2026-09-26.
  - `Mount.reconfigure` writes `profile/override`; `/sandbox` becomes an override.
  - Command-line start options are logged where they differ; nothing is logged for a change
    that matches.
  - *Gate:* every reconfigure leaves a record; an unchanged value leaves none; the log alone
    rebuilds the allowances in force.
  - Built: `ph.session_profile.override` is the one door — it compares through the row's
    model (so `{}` and its defaults are one setting), appends `profile/override`, writes
    the session, and only then calls `Mount.reconfigure`; a record that did not reach disk
    refuses the change (`OverrideNotRecorded`). A gate holds every shipped caller of
    `Mount.reconfigure` to that module. `/sandbox` goes through it and writes no drop-in.
    `open_session` makes one call, `opened`: it records the base, logs this start's
    `--patch` entries that differ (source `cli`), and brings the mount to what the log
    says — each row an override names set once, to its last word, rather than every
    override replayed in turn.
    `rebuilt(base, host, overrides)` composes the base and the overrides in log order.
    `/model`, `session/model`, `--model` and rpc's route are overrides of the session's
    `models` row (`ph.seams.models.move_to`). Settled in the building:
    - **An override is a profile entry** (`{row, entry, source, command}`), not a bare
      config or flag, so the log's environment is the base composed with its overrides —
      a start option that disables or adds a row fits the same record.
    - **Re-applied live until S5.** The daemon still mounts one composition, so without
      putting a session's overrides back at open a `/sandbox allow` would be lost on the
      next start now that it writes no drop-in. Only config patches can be applied to a
      live mount; a start option that disabled or added a row is logged, and applies from
      the log once S5 mounts each root from it.
    - **A route a person names joins the session's list** under a key made from it
      (`route_key`: `fake/fake-9` → `fake-fake-9`), where S1c left it keyless: the
      override has to say what the list now holds, and the footer can name it.
    - **A record that failed to reach disk stays in the session's memory**, as the
      uploads seam's does, and may be written by a later flush — so a later start could
      apply a change this one refused. The direction chosen is that the log never says
      less than was asked.
    - `profile/override-cleared` is left for S7, which has the first writer of it
      (`/profile clear`).
- [x] **S5 — Profiles per session in the daemon.** Done 2026-09-26.
  - `session/new` takes a profile, and the supervisor mounts each root from its log.
  - rpc mode too: it mounts once and serves many sessions, so today one session's
    overrides, brought back by `opened`, are live for the next session on that mount.
    Each session gets a mount of its own, as each daemon root does.
  - *Gate:* two roots on two profiles in one daemon; a root that goes idle and comes back
    keeps its overrides, which fixes the frozen-profile problem.
  - Built: `ph_app.sessions.recorded_environment` reads a stored log's latest base and
    the overrides since, before anything mounts — a line scan that decodes only the
    lines naming a profile record, beside the recorded cwd (`recorded_start` since). `ph_app.profiles
    .session_profile(session_id, requested)` is what every host mounts: the log's
    environment (`rebuilt(base, host, overrides, then=start options)`) when there is
    one, `requested` otherwise. The supervisor composes `requested` from `session/new`'s
    new `profile` (the daemon's `--profile` when empty, refused by name when it does not
    compose), with the daemon's own start options; `RootDescription.profile` says which
    a root runs on; the TUI passes its `--profile` for the sessions it creates. `phern -p
    --session` mounts the same way, and rpc gives each session a mount of its own.
    Settled in the building:
    - **A session's profile is its log's.** `session/new`'s `profile` names the profile a
      *new* session is created on; for one with a base it is not consulted — switching is
      `/profile use` (S7).
    - **Host rows come from the profile the base names, as it composes now**, and from
      `requested` when that one no longer composes: the environment is the log's either
      way, so a renamed or removed named profile costs a session nothing but its host rows.
    - **The daemon's start options reach every root**, whatever profile it was created
      on: they are appended after the log's overrides, and logged where they differ.
    - **`opened`'s bring-to-the-log step stays**, as the fallback for a store the host
      cannot read before mounting (a non-file backend); for a JSONL log it finds nothing
      to do.
    - Still one composition per start, not per root, for the rows the base does not
      hold: a changed named profile's *environment* is S6's question, and until then a
      root mounts its saved base unasked — decision 7's "no" as the default.
- [x] **S6 — A changed named profile on restart, and adopting on purpose.** Done 2026-09-27.
  - The difference, listed and attributed; hold and ask with a person there when the
    person's file changed; report a pH-only change; keep when unattended;
    `phern profiles diff` and `adopt`.
  - *Gate:* the named file edited while the daemon is down holds a root started for the TUI,
    with the list naming the edited setting, and does not hold one started for a schedule. A
    changed plugin default alone asks nothing and is listed. "No" starts it unchanged; "yes"
    saves the new version, logs the switch and re-applies the overrides, then the
    command-line options. `adopt` moves every session on the profile and each keeps its
    overrides.
  - Built: in `ph.session_profile`, `profile/adopted` and `profile/declined` (ignorable)
    and `profile/override-cleared` (required); `fold_environment`, the one reading of a
    log's profile records, live (`logged_environment`) or off disk
    (`ph_app.sessions.recorded_environment`); `switch_base`, the new base and its
    clears in one batch, which `opened` runs first when a version was adopted;
    `profile_change` and `listing` (decision 13's list). `session_profile` returns
    what to mount and the change it found. The supervisor holds a root whose person's
    file moved when the start was asked by a client that declared `asks`
    (`needs-profile-decision`), asks through its desk (`profile/ask`), and on "adopt"
    records the version and starts the root again, its watchers carried over.
    `RootDescription.profileChanges`, a "session profiles" section in `phern agents
    doctor`, and a line in the daemon's log say which roots are behind. `phern
    profiles diff|adopt`; `ph.persistence.stored_session` claims a stored log without
    resuming it, and `session/adopt` records a version in one a daemon holds. The
    TUI's `profile/ask` is a `ConfirmModal`. Settled in the building:
    - **Overrides apply across a base switch until cleared**, where S4's fold read
      only those since the latest base — which had only ever been one base.
      `switch_base` clears the rows whose overrides the new base already says,
      compared through each row's model, so an override written sparse is cleared by
      a base that states every default.
    - **"Asked for by a person" is the client's `asks` capability** on `session/new`
      or `session/attach`. Every other start — a prompt verb, a stored credential, a
      schedule, `phern -p`, rpc — keeps the saved version; `phern -p` says so on
      stderr.
    - **Three answers.** `adopt`; `keep`, recorded, so that version is not asked
      about again (though still listed); and `later` — also Esc — which records
      nothing and asks at the next start.
    - **"Yes" and `adopt` are one mechanism**: both record `profile/adopted`, and the
      next start makes it the base. For a held root that start is at once: it is
      released and mounted again from its log, because a live mount cannot add,
      remove or disable rows. While held, nothing is driven and its children are not
      resumed, since the restart would cut them short.
    - **An override is a whole row's setting**, the loader's rule, so over a row the
      new version changed it still wins, and the new version's change to that row
      does not apply. The listing names each such override ("Still applied over it").
    - **`adopt` writes a stored session under its lease, not resumed**: no repair, no
      reconcile, no `session/resumed`, on a mount of the host's rows alone
      (`log_host`), since nothing runs. A session a daemon holds is written through
      it; one another process holds (a running `phern -p`) is reported not adopted.
    - **A fork's environment is read through its lineage** (found in review): its file
      continues its root's from `seed_length`, and the base is in that prefix, so the
      one-file scan found none and a fork mounted what was asked. `read_stored` is the
      store's one-file read with no mount behind it, which `materialize` walks.
    - ~~Not done: a TUI attaching to a root that kept a changed version says nothing in
      the terminal.~~ Fixed in the follow-ups: the attach notes it (`profileNote`).
      `phern doctor` still does not scan stored sessions.
    - Found in review: a fork's header names its parent (`parent_session`) as a
      child's does, so `record_base` and `opened` passed it over — a fork started as
      a root logged no start options and never switched to a version adopted for it.
      Fixed in S7: a child is told by `origin` (`SessionHeader.is_subagent`).
      ~~Still open: `session/adopt` on a held root leaves the hold and its modal up.~~
      Fixed in the follow-ups: it answers the question (`AskDesk.settle`).
- [x] **S7 — `/profile show | diff | save | use | clear`.** Done 2026-09-27.
  - *Gate:* tune in a session, save, start a headless run on the saved profile, and the two
    environments match.
  - Built: `ph_app.daemon.profile_command`, registered by the supervisor on every root
    beside its ask desk. `show` is `environment_listing` (the base, each override and
    what asked, and each setting that is not the base's), `--full` every row as it
    runs; `diff` is S6's listing against the named profile as it composes now. `save`
    is `ph_app.profiles.save_session`, and `profile/saved {name, path, entries}`
    (ignorable). `use` is `switch_base` with `clear_all`, then `Root.restart_wanted`,
    which `DaemonServer._mutate` honors once the command and its key are durable
    (`Supervisor.restart`, S6's remount, made public). `clear` is
    `ph.session_profile.clear_overrides`: the clears in one batch, written, then the
    live mount brought to them through `_converge` — `opened`'s step 3, factored out.
    Settled in the building:
    - **`/profile` is the host's, not a row's**, as the ask desk is: it saves to the
      host's named-profile store and restarts the host's roots, and commands run only
      in the daemon. So no profile carries or disables it.
    - **A save is the base's own layers and the overrides, as documents** — the
      person's layers as the base recorded them, then every override — over the
      shipped profile the base extends, written sparse. Not the resolved rows: those
      would pin every pH default as it is today and write `${env:...}` settings as
      their values. The file is composed again and kept only when it gives the same
      environment, and taken back otherwise, as `phern profiles fold` does. A name
      is a file name (`PROFILE_NAME`), and an existing file needs `--replace`.
    - **`use` restarts the root**, since a base changes rows a live mount cannot
      follow; it is refused while the agent works or while the root waits on its
      profile question. `use` of the base's own name takes its current version now,
      which is what `/profile diff` offers.
    - **`clear` is live where it can be**: a config override is undone on the running
      mount; one that turned a row on or off restarts the root.
    - **A fork is a session of its own**: `SessionHeader.is_subagent` is `origin == "subagent"`.
    - ~~Not done: a `use` whose new profile fails to mount leaves the session on
      it.~~ Fixed in the follow-ups: `use` is an adoption, and a start that cannot
      mount an adopted version withdraws it and starts on the one it had.
- [x] **S7b — Subagents: an assigned profile, and a model from the list** (items 8, 9).
  Done 2026-09-27.
  - A spawn names a listed model, or its skill does; an assigned profile narrows within the
    parent's ceiling.
  - *Gate:* a skill naming `classify` starts its child on that route; an unlisted key is
    refused; an assigned profile wider than the parent is refused, naming the row.
  - Built: `SubagentRequest.model_key` and `profile`. `SubagentService.resolve_model`
    turns a key — the spawn's, or a named skill's front matter `model:` (`Skill.model`)
    — into a route through the parent's own `ctx.models`, refusing one the list does
    not hold; `subagent/admitted` records `modelKey` beside the route, and a child's
    options carry the key. `resolve_profile` composes a named profile through
    `ctx.named_profiles` (`ph_app.profiles.NamedProfileStore`, provided by
    `runtime.mounted` before the rows) and reads it with
    `ph.seams.subagent_profiles.narrowing` against the parent's mount; the result is
    written into the request before `check_grant`. `ToolRuntime.registrants` says which
    row gave each tool. `rlm.run` and `task` take `model` as a key and a new
    `profile`; the RLM delegation doctrine lists the keys a child can run on.
    Settled in the building:
    - **`model` is a key now, not a model name.** A spawn could name any model string on
      its parent's provider; the list bounds what an agent picks (S1c's settlement). A
      readmitted child keeps the route it was admitted on and is not resolved again.
    - **An assigned profile is read, not mounted**, for what a child can hold less of:
      a row it runs that the parent does not is refused by name; the tools of rows it
      does not run are taken away; skills by the `skills-progressive` paths; its models
      default, which the parent must list; and a read-only sandbox default. Everything
      else it says is the parent's, since the child runs on the parent's mount.
    - **A ceiling, not defaults**, unlike a preset: tools or skills named beside it may
      narrow further, and naming more than it gives is refused, as is `access="write"`
      under a read-only one.
    - **The host composes it** (`ctx.named_profiles`): ph-core has no named-profile
      store, and `runtime.mounted` is the one door every phern host mounts through.
    - Not done: a narrower sandbox than the parent's other than read-only. Fixed in
      the follow-ups for writable directories (`SandboxSeam.restrict_paths`) and for a
      dropped row's skills (`SkillService.registrants`). **Still open: hosts** — one
      egress proxy serves every agent, so a profile with fewer hosts than its
      parent is refused rather than given its parent's.
- [x] **S8 — Reading the audit.** The environment at any seq, in the CLI and the trajectory
  view, and the skill-read records (decision 12). Done 2026-09-27.
  - Built: `phern profiles session <id> [--at SEQ] [--full]` — the fold of the log's
    prefix, read through its lineage (`materialize` over `read_stored`, so a fork
    answers too): `environment_listing` and the skills read by then, or every row as
    it ran. The trajectory view gives each `profile/*` record a summary
    (`record_summary`) and, as its detail, the environment it leads to.
    `ph.seams.skills.record_read` writes an ignorable `skill/read {name, version,
    path, sha256, via}` for the `skill` tool's read (`via: tool`) and for a named
    skill's body put in a child's prompt at a spawn (`via: brief`).
    Settled in the building:
    - **`phern profiles session` prints the environment, not the base record**: the
      base, the overrides and what they change, as `/profile show` does, with
      `--full` for every row. The raw `profile/base` is in the log for anyone who
      wants the record itself.
    - **The environment is shown where it changes**: each `profile/*` record in the
      trajectory carries the environment from that point, rather than every record
      repeating it.
    - **A brief is a read**: the body a child is prompted with is recorded in its
      parent's log, where the admission that caused it is.
    - Not done: the hash is of the body as read, before a skill's declared inputs
      are filled in — two reads with different arguments hash the same, which is
      what "the same instructions" means here.
- [x] **Follow-ups — what the steps left open.** Done 2026-09-27, except as noted.
  - **Every base change after the first is an adoption applied at a start**, so
    `/profile use`, a "yes" and `phern profiles adopt` share one rule, and one rollback:
    a start the loader refuses the adopted version at withdraws it —
    `profile/withdrawn {…, reason}` (`withdraw_adoption`), required, read by the fold
    as a decline so it is not offered again — and starts on the version it had. The
    reason reaches the command's reply, the attach (`RootDescription.profileNote`),
    `phern -p` and rpc (one rule for every host, `runtime.mount_session`) and
    `/profile show`. Only the loader's
    refusals (`LoaderError`, `MountRefusal`, a config's `ValidationError`) withdraw;
    a disk or a bug fails the start and leaves the adoption. An adoption can carry
    `clear: true` (`/profile use --clear`).
  - **A mutation's `after` hook** (`Mutation.after`) replaces the generic
    `restart_wanted` check in `_mutate`: the command row's hook restarts the root once
    the command is durable, and also when it failed, so a request is never left for
    the next command.
  - **`session/adopt` settles the held root's question** (`AskDesk.settle`), which
    tells every front end, so its modal closes.
  - **The TUI notes a kept version on attach**, in the daemon's own sentence
    (`RootDescription.profileNote`, `kept_note` with `/profile`'s commands), so a
    terminal and the command line word it once — whose the settings are included.
  - **One per-scope table for the three registries that narrow per agent**
    (`ph.cordis.ScopedTable`, `ScopedEntries`): the tool registry's layers, the skill
    registry's restrictions — whose release now also refreshes the catalog when a
    child's scope unwinds — and the sandbox's writable-directory limits.
  - **Each scope knows its row** (`Context.row_id`, stamped by the loader and
    inherited), so `ToolRuntime.registrants` reads it rather than keeping its own
    table, and `SkillService.registrants` exists.
  - **A plugin declares how a child holds less of it** (`plugin(..., narrows=)`,
    `ph.cordis.child_limit`), typed against its config model; `models`,
    `sandbox-policy`, `skills-progressive` and `sandbox-allow` are the four, and
    `ph.seams.subagent_profiles` names none of them.
  - **A child binds only the writable directories its profile keeps**
    (`SandboxSeam.restrict_paths`, the admission's `paths`), in the prompt boundary
    too (`allowed_paths_of(ctx, agent)`).
  - Deferred: fewer hosts for a child (a per-agent egress list), and the
    named-profile store as a deployment row.
