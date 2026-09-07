# Writing pH for Linux *and* macOS

**Status:** written 2026-09-07, from what `a20ac5e`…`HEAD` turned up while making the
suite pass on a Mac for the first time.

pH was developed on Linux and CI ran a two-OS matrix from Phase 0. That matrix was
red often enough that nobody read it, so the first serious macOS run found **~170
failures, 53 errors, and 14 more behind those** — and *not one* of them was a thing
macOS cannot do. Every single one was a place where the code, a test, or a
constant had quietly assumed a Linux property.

That is the headline, and it is why this file exists rather than a `skipif`:

> A test that fails on one platform is a claim about your code until you have
> proved otherwise. Every time we assumed "platform gap" here, we were wrong.

Twice during this work a list of "known macOS failures" was written down with a
plausible reason beside each entry. Both lists were later deleted in full, because
investigating the reasons found defects instead — including one in shipped product
behaviour (`ph agents attach --until-idle` printed a different thing depending on
which side of a race it landed on). The allowlist was not neutral bookkeeping. It
was hiding bugs.

---

## The differences, and what each one broke

### 1. Unix socket paths are shorter than you think, and `$TMPDIR` is longer

`sun_path` is **104 bytes on Darwin, 108 on Linux** — a hard limit in the kernel
struct, not a filesystem limit. Linux's `$TMPDIR` is `/tmp` (4 bytes). macOS gives
every process a per-user one:

```
/var/folders/n4/k878bj0543l13kdslnxxnpch0000gr/T/     49 bytes, before you add anything
```

pytest then appends `pytest-of-<user>/pytest-<n>/<test-name>0/`, and a daemon
socket lands at **126 bytes**. Every bind fails with `AF_UNIX path too long`.

**Cost:** ~170 failures and ~53 errors across the daemon, TUI, web and agents-CLI
suites. All of them looked like daemon bugs. None of them were.

**Fix:** `test.sh` exports `TMPDIR=/tmp/ph-test-<uid>` before pytest — 87 bytes for
the same socket. This is a *fix*, not a workaround: the tests were correct and the
harness was giving them a path no kernel would accept.

**Rule:** any code that binds a unix socket under a caller-supplied directory needs
a fallback for a path that will not fit. `ph.seams.sandbox_local.egress_socket_path`
already had one (`SOCKET_PATH_MAX = 100`, falling back to a private temp dir); the
daemon and the test fixtures did not.

### 2. There is no systemd, and "is this Linux" is the wrong question anyway

`ph.lingering` answers whether a daemon outlives logout. It began with:

```python
if sys.platform != "linux":
    return "not-applicable"
```

That short-circuited **before** reading the marker directory, so twelve tests that
staged a complete simulated logind host — `$XDG_RUNTIME_DIR`, a patched
`LINGER_DIR`, marker files — were answered by the platform name and never by the
evidence they had arranged.

The guard was also wrong *on Linux*: a host without systemd (Alpine, a slim
container, sysvinit) was told to run a `loginctl` it does not have.

**Fix:** read the evidence. `logind_present()` stats `/run/systemd/system`, which is
systemd's own `sd_booted(3)` test.

**Two traps worth naming**, because the first attempt hit both:

- **Installed is not booted.** `/var/lib/systemd` is created by `systemd-timesyncd`
  and friends, and survives on a host systemd did not boot. Keying off it gives the
  same wrong advice to a narrower set of hosts. `/run/systemd/system` exists only
  where systemd is PID 1.
- **Don't infer one fact from another fact's constant.** Deriving "systemd exists"
  from `LINGER_DIR.parent` made the no-logind branch *impossible to test*: a test
  points `LINGER_DIR` at its own temp directory, which always has a parent that
  exists. The branch had zero coverage until the presence check got its own
  patchable constant (`SYSTEMD_RUN_DIR`) and the fixture gained a `logind=False`
  mode.

**Rule:** prefer a capability probe over a platform name. `sys.platform` answers
"which kernel", and you almost always want "is this facility here" — which is also
the only form a test can stage.

### 3. Seatbelt is a deny-list; bwrap is a namespace. They are not the same shape

macOS confinement is `sandbox-exec` with a generated SBPL profile. Linux is
`bwrap`. Four consequences, each of which broke something:

**No PID namespace, so helpers leak.** `bwrap --unshare-pid` makes a backgrounded
`socat` shim die with the command. Seatbelt unshares nothing, so the shim
*outlived* every command — one from a test run was still holding port 3128 an hour
later, and the next command's bind found it taken. This is why the macOS egress
door is the proxy's own loopback port and **needs no `socat` at all**. Each backend
now declares its door (`needs_loopback`) and owns its own probe.

**No network namespace, so "no network" is spelled differently.** Under `bwrap`,
`--unshare-net` means nothing inside can reach anything — enforcement is structural.
Under Seatbelt the command shares the host's loopback, and the profile must *deny*
everything and allow exactly one remote back. Tests that counted interfaces in
`/proc/net/dev` were asking a namespace question that has no macOS answer; they now
ask what both must answer alike — can a confined command reach a listener this
process opened?

**Signals reach the command.** `bwrap` dies of `SIGINT` itself and takes the
namespace with it, so signalling it *destroys* rather than interrupts.
`sandbox-exec` execs its target, so a signal lands normally. `ConfinedArgv.forwards_signals`
is declared per backend rather than derived from "was I confined", because deriving
it would silently drop a working cancel route on macOS.

**The kernel refuses in different words.** bwrap/glibc says `Read-only file system`
and `Network is unreachable`. Seatbelt says `Operation not permitted` for *both*
boundaries, and the only way to tell them apart is whether the line carries a path.
Each backend owns its own signature table; neither reads the other's.

### 4. Symlinked system directories: `/var` is `/private/var`

On macOS `/var`, `/tmp` and `/etc` are symlinks into `/private`. Seatbelt matches
`subpath` rules against the path the **kernel resolves**, so a workspace under
`$TMPDIR` named as `/var/folders/…` was refused *its own writes*.

The tempting fix — resolve the path in the backend that noticed — is wrong, and we
wrote it before catching it. It makes the *enforced* boundary `/private/var/…` while
the *prompt* boundary that `permissions-fs` draws from the same set stays `/var/…`,
so a person gets asked about a write the kernel permits. That is the exact drift the
"one definition of the writable set" rule exists to prevent.

**Fix:** canonicalise where roots are *minted* — `ph.paths.canonical`, applied to the
three `$PH_*` roots, `default_home_path`, the workspace seam's `base` and `scratch`,
and `sandbox-allow`'s directories. Every consumer then reads one spelling.

**Rule:** if two layers compare paths, they must agree on the spelling, and the
place to settle it is where the path is created — never in the layer that happened
to notice.

### 5. Socket chunk sizes differ, and they can hide a broken limit

`net.local.stream.recvspace` is **8 KiB on macOS, 64 KiB on Linux**.

anyio's `receive_until` searches for the delimiter *before* it tests the buffer
against `max_bytes`. So an over-length frame is refused only when its newline has
not yet arrived — which depends entirely on how much the socket hands over per read.

`MAX_LINE` was documented as "how long one frame may be" and was not a cap at all. A
6 MiB attachment sailed through on Linux and closed the connection on macOS. A test
written to pin a *named* refusal passed on one kernel while proving the opposite on
the other.

**Fix:** `read_frames` re-checks the length of what it returned.
`test_daemon_framing.py` parametrises the chunk size so both kernels' behaviour is
asserted on either platform.

**Rule:** a limit enforced by a library's scan order is not enforced. If you
document a bound, test it at the bound — and if the behaviour can turn on a buffer
size, make the buffer size a test parameter.

### 6. Resource limits macOS simply refuses, and a report that could not be read

macOS refuses `RLIMIT_AS` outright (`ValueError: current limit exceeds maximum
limit`). The guest reported the soft limit still in force — `RLIM_INFINITY`, 2^63-1
— and the host's codec refuses integers above 2^53 as non-lossless, so it **dropped
the entire `boot-ack` frame as junk**. Every kernel start then waited out a 30-60s
timeout and reported "the runtime did not report ready", with nothing to quote.

No RLM kernel had *ever* booted on macOS. `test_kernel.py` failed before a sandbox
was involved.

Three separate defects, one symptom:

- The guest reported a number that is not a limit. It now reports `None` for a limit
  it could not apply.
- The **codec's integer rule was a whole-frame veto**. A `parse_int` hook rejected a
  big integer *anywhere* in the line, including inside fields the spec declares
  `obj`/`any` and does not inspect. Same bug bit every platform: a cell ending in
  `2**60` never settled. The bound now lives in field coercion, where the other
  shape rules are.
- The host **looped silently** on a frame it could not read. Before `boot-ack`
  nothing model-written has run, so there is no hostile peer to be tolerant of;
  anything that is not `boot-ack` or `fault` is now a fault that quotes the line.

**Rule:** when a subsystem hangs, suspect the *report* as well as the thing being
reported. And a diagnostic that says only "it did not start" is a defect — quote
what the child actually said.

### 7. Terminal width and hard wrapping mangle the things you need to copy

Rich's default folds a line at the console width with a hard newline — **inside a
word** when the word is longer than what is left of the line. Under a test runner
the width is 80. A macOS temp path is about a dozen characters longer than the same
path on Linux, and that was the exact margin by which a filename stopped appearing
in the refusal that named it.

**Fix:** `soft_wrap=True` on both `Console` objects in `ph_app/console.py`, beside
`highlight=False` and for the same reason — a console setting applied to half the
CLI is how the two halves come to disagree.

**Still true and worth knowing:** `soft_wrap` is a print-time setting and does *not*
reach `Table` cells, so a path printed as a table value still folds.

**Rule:** anything a person will copy — a path, an id, a command to run — must not
be folded. Set it on the console, not at the call site.

### 8. `ruff format` rewrites fenced code inside Markdown

Not a platform difference, but it is what kept CI red throughout and so belongs in
the same story. `ruff format .` formats Python inside ```` ```python ```` blocks in
Markdown, and twelve documentation files failed on a clean checkout. That is why
`plans/` and `sources/` are in `extend-exclude`: reformatting a *quoted excerpt*
would silently edit the record it quotes.

The twelve were pH's own code samples, so they were simply formatted — one of them,
DESIGN.md's excerpt of the family reach rule, had been hand-compacted and now matches
the shape of the source it cites.

**Rule:** a formatting complaint is never a platform gap. It is deterministic and
one command from being gone, so it must never go on a tolerated-failure list.

---

## What to do when a test fails on one platform only

1. **Assume it is your bug.** The base rate in this repository is 14 out of 14.
2. **Find the mechanism before you find a workaround.** The mechanisms that actually
   came up: path length, `$TMPDIR` shape, symlinked system directories, socket buffer
   size, console width, missing resource limits, absent init system, namespace vs
   deny-list confinement.
3. **Ask whether the test is staging something the code refuses to look at.** Twelve
   failures were one guard that answered before reading the fixture's evidence.
4. **Check whether the "passing" platform is passing for a good reason.** Three of
   these were latent everywhere and merely *visible* on one kernel — the codec veto,
   the frame cap, the `--until-idle` race. macOS was the better test rig.
5. **If you must record a gap, make it operative.** Name the mechanism and what
   would change the answer. "macOS is different" is not a reason; "macOS has no
   `loginctl`, and here is the probe that would tell us" is.

`test.sh` keeps a per-platform gap list for this purpose, with the matcher and the
"this gap now passes, delete it" check. **Both arms are empty**, and the bar for
adding one is a mechanism of the kind the fourteen turned out not to have.

## See also

`test.sh` · `docs/seams/sandbox.md` (the two backends and their doors) ·
`docs/seams/workspace.md` (canonical roots) · `packages/ph-rlm/tests/test_boot_report.py`
· `packages/ph-app/tests/test_daemon_framing.py` · plan rows P6-40, P6-41
