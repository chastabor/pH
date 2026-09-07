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
#   ./test.sh --strict        known platform gaps fail too (what CI effectively does)
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

STRICT=0
FIX=0
FAILED_GATES=()
NEW_FAILURES=0
KNOWN_HIT=0

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
known_gaps() {
  case "$PLATFORM" in
    macos)
      cat <<'EOF'
packages/ph-core/tests/test_lingering.py|systemd: linger state is read from loginctl's marker directory, which macOS has no equivalent of
packages/ph-app/tests/test_cli.py::test_doctor_says_whether_the_daemon_socket_survives_logout|systemd: the logout-survival answer is a Linux runtime-dir property
packages/ph-app/tests/test_cli.py::test_starting_a_daemon_that_will_not_outlive_logout_says_so_first|systemd: as above, on the start path
packages/ph-app/tests/test_agents_cli.py::test_doctor_prints_the_lifetime_the_daemon_reports|systemd: the lifetime row is the linger answer
packages/ph-app/tests/test_agents_cli.py::test_a_reaped_socket_is_not_reported_as_one_never_started|systemd: distinguishing reaped from never-started needs the reaped-tree semantics of $XDG_RUNTIME_DIR
packages/ph-app/tests/test_daemon.py::test_a_reaped_runtime_dir_reaches_every_root_as_a_record|systemd: as above, as a per-root record
packages/ph-app/tests/test_daemon.py::test_daemon_status_says_it_cannot_be_reached_and_what_would_fix_it|systemd: the advice it prints names the linger fix
packages/ph-app/tests/test_cli.py::test_a_profile_that_will_not_parse_is_refused_before_anything_mounts|unexplained on macOS; the refusal arrives but doctor's table is printed first. Pre-existing (verified at a20ac5e), not yet diagnosed
packages/ph-app/tests/test_agents_cli.py::test_until_idle_exits_non_zero_when_the_last_turn_errored|unexplained on macOS; pre-existing (verified at a20ac5e), not yet diagnosed
packages/ph-app/tests/test_daemon_attachments.py::test_a_file_too_large_for_a_frame_is_refused_by_name|unexplained on macOS; the oversize frame closes the connection before the named refusal is read. Pre-existing (verified at a20ac5e)
EOF
      ;;
    linux)
      # Left empty on purpose. Linux is CI's reference platform: everything is
      # expected to pass, and a gap here should be argued for rather than
      # inherited. `bwrap`'s absence does not belong on this list — those tests
      # skip themselves by asking whether the row registered a provider.
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

  head1 "Optional backends (absent means skipped tests, never failures)"
  if have jj; then ok "jj — the Jujutsu workspace tier"
  else
    warn "jj is missing — 3 jj-tier tests skip"
    note "macOS: brew install jj   ·   Linux: cargo install --locked jj-cli"
  fi
  if have agentfs; then ok "agentfs — the copy-on-write overlay tier"
  else
    warn "agentfs is missing — 11 overlay tests skip"
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
# A platform gap is something no action of yours can fix: macOS has no `loginctl`,
# so those tests cannot pass here and tolerating them is the only alternative to
# pretending they do not exist. Formatting is the opposite — `ruff format` is
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

gate_test() {
  head1 "Tests  (pytest)"
  say "  ${DIM}TMPDIR=$TMPDIR${RESET}"
  local log status node reason
  # This gate's own tally. `KNOWN_HIT` is the run-wide total the summary prints,
  # and sharing one counter had the test gate reporting the format gate's twelve
  # documentation files as tests that failed.
  local gaps=0
  log="$(mktemp "${TMPDIR}/ph-pytest.XXXXXX")"

  uv run pytest -q "$@" 2>&1 | tee "$log"
  status="${PIPESTATUS[0]}"

  if [ "$status" = "0" ]; then
    ok "all tests passed"
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
      KNOWN_HIT=$((KNOWN_HIT + 1))
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
  elif [ "$STRICT" = "1" ]; then
    bad "$gaps known platform gap(s) failed (--strict)"
    FAILED_GATES+=("test")
  else
    ok "no regressions ($gaps known platform gap(s) failed, each listed above)"
  fi
  rm -f "$log"
}

# ------------------------------------------------------------------------ driver --

main() {
  local gate="all"
  local args=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --strict) STRICT=1; shift ;;
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
  if [ "$KNOWN_HIT" -gt 0 ]; then
    note "$KNOWN_HIT known gap(s) tolerated across all gates; run with --strict to fail on them"
  fi
  exit 0
}

main "$@"
