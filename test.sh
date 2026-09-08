#!/usr/bin/env bash
#
# Run pH's gates on macOS or Linux, and tell the truth about the difference.
#
# The four gates are CI's, in CI's order and with CI's commands (.github/workflows
# /ci.yml): lint, format, types, test. What this script adds is everything that
# has to be true *around* them for a local run to mean the same thing as a CI run
# on the other platform:
#
#   * a TMPDIR short enough for a unix socket (macOS; see `prepare_tmpdir`),
#   * a report of which optional backends are installed, and what each missing
#     one costs in coverage rather than in failures,
#   * a per-platform list of tests that are known not to run here, so a green
#     result means "no regressions" instead of "no failures, some of which I have
#     silently stopped counting".
#
# The last one is the reason this file exists rather than a one-line alias. A
# suite that is always a little bit red stops being read — CI's own comment about
# dropping Windows from the matrix makes exactly this argument — so the gaps are
# enumerated, with a reason each, and the script fails on anything that is not on
# the list. It also tells you when a listed gap has started passing, because a
# baseline nobody prunes is a baseline that quietly hides the next regression.
#
# Usage:
#   ./test.sh                 every gate
#   ./test.sh doctor          just the environment report
#   ./test.sh lint|format|types|test
#   ./test.sh --fix           auto-correct what lint and format can, then gate
#   ./test.sh format --fix    the same, for the format gate alone
#   ./test.sh test -k pattern anything after the gate goes to pytest
#   ./test.sh test --cov      with coverage, as CI runs it
#
# Lint and format are *checks* by default rather than rewrites, because a command
# whose name is "test" should not quietly edit your working tree while you are
# reading a diff. `--fix` is the one keystroke that says you want it to.
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

# ---------------------------------------------------------------- presentation --

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
  DIM=$'\033[2m'; RESET=$'\033[0m'
else
  BOLD=''; RED=''; GREEN=''; YELLOW=''; DIM=''; RESET=''
fi

say()  { printf '%s\n' "$*"; }
head1() { printf '\n%s==> %s%s\n' "$BOLD" "$*" "$RESET"; }
ok()   { printf '  %sok%s      %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '  %swarn%s    %s\n' "$YELLOW" "$RESET" "$*"; }
bad()  { printf '  %sFAIL%s    %s\n' "$RED" "$RESET" "$*"; }
note() { printf '  %s%s%s\n' "$DIM" "$*" "$RESET"; }

# --------------------------------------------------------------------- platform --

OS="$(uname -s)"
case "$OS" in
  Darwin) PLATFORM=macos ;;
  Linux)  PLATFORM=linux ;;
  *)      PLATFORM=other ;;
esac

FIX=0
FAILED_GATES=()
NEW_FAILURES=0

# ------------------------------------------------------------------- the TMPDIR --

# **The single most important line in this file on macOS.**
#
# A unix socket path is capped at `sun_path` — 104 bytes on Darwin, 108 on Linux —
# and the daemon, the egress proxy and several tests bind one under pytest's
# `tmp_path`. macOS hands every process a per-user `$TMPDIR` like
# `/var/folders/n4/k878bj0543l13kdslnxxnpch0000gr/T/`, which is 49 bytes before
# pytest has added `pytest-of-<user>/pytest-<n>/<test-name>0/`; a representative
# daemon socket lands at **126 bytes** and the bind fails with `AF_UNIX path too
# long`. Measured: that single cause accounts for ~170 failures and ~53 errors
# across `packages/ph-app` — the daemon, TUI, web and agents-CLI suites — none of
# which are real. With a short TMPDIR the same path is 87 bytes and every one of
# them passes.
#
# So this is a fix, not a workaround, and it is why `./test.sh` on a Mac and CI on
# Linux agree. `/tmp/ph-test-<uid>` is per-user and `0700` for the reason
# `ph.paths` gives about its own tier-3 fallback: a shared `/tmp` directory
# somebody else can write is not one to adopt.
prepare_tmpdir() {
  local dir="/tmp/ph-test-$(id -u)"
  if ! mkdir -p "$dir" 2>/dev/null; then
    warn "could not create $dir; leaving TMPDIR as it is"
    return
  fi
  chmod 700 "$dir" 2>/dev/null || true
  export TMPDIR="$dir"
  # `resolve_roots` and every workspace root are canonical (ph.paths.canonical),
  # so handing them a symlinked TMPDIR would have the suite comparing two
  # spellings of one directory — the drift the canonical roots exist to end.
  if command -v python3 >/dev/null 2>&1; then
    TMPDIR="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$dir" 2>/dev/null || printf '%s' "$dir")"
    export TMPDIR
  fi
}

# ---------------------------------------------------------------- the gap lists --

# Tests that do not run on this platform, each with the reason. **Prefix match on
# the pytest node id**, so a whole file can be named without listing every test in
# it. Anything failing that is not matched here is a regression and fails the run.
#
# **Both lists are empty, and that took work to earn.** The first version of this
# script carried fourteen macOS entries. Investigating them found no platform
# limitation among them — twelve were one `sys.platform != "linux"` guard in
# `ph.lingering` that answered before reading the evidence a test had staged; one
# was `fail()` hard-wrapping a path inside a word at 80 columns, where a macOS temp
# path is a dozen characters longer than Linux's; one was `--until-idle` printing
# its status line in one of two race outcomes; one was a test payload 150 bytes over
# the frame ceiling that Linux forgave by handing anyio bigger socket chunks. Each is
# fixed where it lived. A gap added here needs a reason of the kind those turned out
# not to have.
known_gaps() {
  case "$PLATFORM" in
    macos) : ;;
    linux)
      # Linux is CI's reference platform, so a gap here needs arguing for rather
      # than inheriting. The discriminator, kept because it is the concrete shape
      # of the bar: `bwrap`'s absence does *not* belong on this list — those tests
      # skip themselves by asking whether the row registered a provider, which is
      # a test declining to run, not a test failing.
      : ;;
  esac
}

reason_for() {
  # First gap line whose id is a prefix of "$1", or empty.
  local node="$1" line id
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    id="${line%%|*}"
    case "$node" in "$id"*) printf '%s' "${line#*|}"; return 0 ;; esac
  done <<EOF
$(known_gaps)
EOF
  return 1
}

# ------------------------------------------------------------------- the doctor --

have() { command -v "$1" >/dev/null 2>&1; }

report_env() {
  head1 "Environment"
  say "  platform    $OS ($(uname -m))"
  say "  TMPDIR      $TMPDIR"
  if have uv; then
    say "  uv          $(uv --version 2>/dev/null)"
  else
    bad "uv is not installed — everything below needs it (https://docs.astral.sh/uv/)"
    exit 1
  fi
  say "  python      $(uv run python -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo '?')"

  head1 "Required"
  for tool in git; do
    if have "$tool"; then ok "$tool"; else bad "$tool is missing"; fi
  done

  head1 "Confinement backend (the sandbox tier)"
  case "$PLATFORM" in
    macos)
      if have sandbox-exec; then
        ok "sandbox-exec — the Seatbelt backend, verified on macOS 26.6"
        note "socat is NOT needed here: the macOS egress door is a loopback port, not a shim."
      else
        warn "sandbox-exec is missing, which is unusual on macOS; sandbox tests will skip"
      fi
      ;;
    linux)
      if have bwrap; then
        ok "bwrap"
        # Ubuntu 23.10+ defaults `kernel.apparmor_restrict_unprivileged_userns=1`,
        # and an unprofiled bwrap is not setuid, so it dies with "setting up uid
        # map: Permission denied" and the row declines the tier rather than
        # claiming it. That decline is correct, and it is also invisible unless
        # somebody says this out loud.
        local restricted=""
        [ -r /proc/sys/kernel/apparmor_restrict_unprivileged_userns ] &&
          restricted="$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null)"
        if [ "$restricted" = "1" ] && [ ! -e /etc/apparmor.d/bwrap ]; then
          warn "bwrap is present but unprofiled under apparmor_restrict_unprivileged_userns=1"
          note "the sandbox tier will decline and its tests will skip. Add /etc/apparmor.d/bwrap to enable them."
        fi
      else
        warn "bwrap is missing — the sandbox tier declines and its tests skip"
        note "install it with: sudo apt install bubblewrap"
      fi
      if have socat; then
        ok "socat — the egress shim bwrap's network door needs"
      else
        warn "socat is missing — under bwrap, 'allowlist' means no network and the egress tests skip"
      fi
      ;;
  esac

  # **No counts here.** This runs before pytest, so a number written at this
  # point is one nobody can check — and the two that used to be here (3 and 11)
  # were both wrong by the time anybody read them: `test_workspace_jj.py` alone
  # skips two dozen. What each missing tool actually costs is counted from the
  # run and printed by `report_skips` below, where it is a measurement rather
  # than a claim.
  head1 "Optional backends (absent means skipped tests, never failures)"
  if have jj; then ok "jj — the Jujutsu workspace tier"
  else
    warn "jj is missing — its tier's tests skip; the run counts them"
    note "macOS: brew install jj   ·   Linux: cargo install --locked jj-cli"
  fi
  if have agentfs; then ok "agentfs — the copy-on-write overlay tier"
  else
    warn "agentfs is missing — the overlay tier's tests skip; the run counts them"
    note "there is no packaged build; the overlay tier simply declines without it"
  fi
  # Turso is a *Python* dependency (pyturso), not a CLI — `uv sync` provides it,
  # and a missing CLI is not what would break it. Said explicitly because it is
  # easy to assume otherwise from the row's name.
  if uv run python -c 'import turso' >/dev/null 2>&1; then
    ok "pyturso — the Turso persistence backend (a Python package; no CLI to install)"
  else
    warn "pyturso did not import — run: uv sync"
  fi
}

# --------------------------------------------------------------------- the gates --

gate_lint() {
  if [ "$FIX" = "1" ]; then
    head1 "Lint  (ruff check --fix .)"
    uv run ruff check --fix . && ok "clean"
    # Not every rule is auto-fixable, so this still gates on the result.
    uv run ruff check . >/dev/null 2>&1 || { bad "lint findings remain that ruff cannot fix"; FAILED_GATES+=("lint"); }
    return
  fi
  head1 "Lint  (ruff check .)"
  if uv run ruff check .; then ok "clean"; else bad "lint"; FAILED_GATES+=("lint"); fi
}

# **No allowlist here, unlike `known_gaps`, and the difference is the point.**
#
# A platform gap would be something no action of yours can fix — and note that the
# fourteen macOS failures this script first shipped with all turned out *not* to be
# that, which is why `known_gaps` is empty. Formatting is further still — `ruff format` is
# deterministic and idempotent, so every complaint it makes is one command away
# from being gone for good. An allowlist there would not be recording a fact about
# the platform, it would be deferring a fix forever and calling it a fact. So this
# gate is binary: clean, or fail with the command that fixes it.
#
# (The twelve documentation files this used to excuse were simply formatted. They
# were pH's own code samples, and normalising them made DESIGN.md's quoted reach
# rule match the shape of the source it cites, rather than a hand-compacted
# paraphrase of it.)
gate_format() {
  if [ "$FIX" = "1" ]; then
    head1 "Format  (ruff format .)"
    uv run ruff format . && ok "formatted"
    return
  fi
  head1 "Format  (ruff format --check .)"
  if uv run ruff format --check .; then
    ok "clean"
  else
    bad "unformatted files, listed above"
    note "fix them with: ./test.sh --fix    (or: uv run ruff format .)"
    FAILED_GATES+=("format")
  fi
}

gate_types() {
  head1 "Types  (mypy)"
  if uv run mypy; then ok "clean"; else bad "types"; FAILED_GATES+=("types"); fi
}

# What was skipped, grouped by the reason the test gave for skipping.
#
# pytest already prints one `SKIPPED [n] <file>:<line>: <reason>` line per skip
# (the `-ra` in `[tool.pytest.ini_options]`), so the reasons are the suite's own
# words — nothing here decides what a skip means or how many there are. Grouped
# because 61 individual lines is a wall and "24 · the jj tier needs jj" is the
# fact: which capability this host is missing, and what it costs.
report_skips() {
  local log="$1" rows
  # `SKIPPED [n] path:line: reason` → `n<TAB>reason`. The location token has no
  # spaces and ends in a colon, so it can be consumed whole; anything that does
  # not match that shape is kept verbatim rather than dropped.
  rows="$(
    grep -E '^SKIPPED \[[0-9]+\]' "$log" |
      sed -E 's/^SKIPPED \[([0-9]+)\] [^ ]+ (.*)$/\1\t\2/' |
      awk -F'\t' 'NF==2 {n[$2]+=$1; next} {n[$0]+=1} END {for (r in n) printf "%d\t%s\n", n[r], r}' |
      sort -rn
  )"
  [ -n "$rows" ] || return 0
  local total
  total="$(printf '%s\n' "$rows" | awk -F'\t' '{t+=$1} END {print t+0}')"
  note "$total skipped, by reason:"
  printf '%s\n' "$rows" | while IFS=$'\t' read -r count reason; do
    printf '  %s%6s  %s%s\n' "$DIM" "$count" "$reason" "$RESET"
  done
}

gate_test() {
  head1 "Tests  (pytest)"
  say "  ${DIM}TMPDIR=$TMPDIR${RESET}"
  local log status node reason
  local gaps=0
  log="$(mktemp "${TMPDIR}/ph-pytest.XXXXXX")"

  uv run pytest -q "$@" 2>&1 | tee "$log"
  status="${PIPESTATUS[0]}"

  if [ "$status" = "0" ]; then
    ok "all tests passed"
    report_skips "$log"
    # A gap that has started passing is a line to delete, and saying so is what
    # keeps the list from rotting into a place regressions hide.
    while IFS= read -r line; do
      [ -n "$line" ] || continue
      warn "known gap now passes, remove it from known_gaps(): ${line%%|*}"
    done <<EOF
$(known_gaps)
EOF
    rm -f "$log"
    return
  fi

  while IFS= read -r node; do
    [ -n "$node" ] || continue
    if reason="$(reason_for "$node")"; then
      gaps=$((gaps + 1))
      note "known: $node"
      note "       $reason"
    else
      bad "$node"
      NEW_FAILURES=$((NEW_FAILURES + 1))
    fi
  done < <(grep -E '^FAILED ' "$log" | sed 's/^FAILED //' | sed 's/ - .*//' | sort -u)

  # An error is a test that could not even run; never quietly tolerated.
  #
  # Read from pytest's own count line rather than by grepping `^ERROR`, which
  # matches a *captured log record* at ERROR level — the daemon suite emits one
  # ("... is no longer this daemon's socket") on a run with no errors at all, and
  # counting it turned a clean result into a failing one.
  local errors
  errors="$(grep -oE '[0-9]+ error' "$log" | tail -1 | grep -oE '^[0-9]+')"
  if [ -n "$errors" ] && [ "$errors" -gt 0 ]; then
    bad "$errors collection/setup error(s) — see the output above"
    grep -oE '^ERROR [^[:space:]]+::[^[:space:]]+' "$log" | sed 's/^ERROR /        /' | sort -u
    NEW_FAILURES=$((NEW_FAILURES + errors))
  fi

  if [ "$NEW_FAILURES" -gt 0 ]; then
    FAILED_GATES+=("test")
  elif [ "$gaps" -gt 0 ]; then
    ok "no regressions ($gaps known platform gap(s) failed, each listed above)"
  else
    ok "no regressions"
  fi
  report_skips "$log"
  rm -f "$log"
}

# ------------------------------------------------------------------------ driver --

main() {
  local gate="all"
  local args=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --fix) FIX=1; shift ;;
      -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
      doctor|lint|format|types|test|all) gate="$1"; shift; args=("$@"); break ;;
      *) args+=("$1"); shift ;;
    esac
  done

  prepare_tmpdir

  case "$gate" in
    doctor) report_env ;;
    lint)   gate_lint ;;
    format) gate_format ;;
    types)  gate_types ;;
    test)   report_env; gate_test "${args[@]+"${args[@]}"}" ;;
    all)
      report_env
      gate_lint
      gate_format
      gate_types
      gate_test "${args[@]+"${args[@]}"}"
      ;;
  esac

  head1 "Summary"
  if [ "${#FAILED_GATES[@]}" -gt 0 ]; then
    bad "failed: ${FAILED_GATES[*]}"
    exit 1
  fi
  ok "everything green on $PLATFORM"
  exit 0
}

main "$@"
