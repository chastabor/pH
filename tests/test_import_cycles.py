"""N3: every shipped module imports first, so moving an import to the top is safe.

An import inside a function runs only when its function does, so a misspelled name
or a cycle waits for the one process that happens to call it in that order. N3 moved
those to module top, and ruff's `PLC0415` now holds shipped code to it: each import
that stays in a function says why in a comment above it and carries
`# noqa: PLC0415` (see `pyproject.toml`).

A move to the top can open a cycle, which ruff cannot see. This is what does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from workspace_layout import parsed_modules, workspace_packages

PROBE = """
import importlib, json, sys
names, tops = json.loads(sys.argv[1]), set(json.loads(sys.argv[2]))
failed = {}
for name in names:
    for loaded in [module for module in sys.modules if module.split(".")[0] in tops]:
        del sys.modules[loaded]
    try:
        importlib.import_module(name)
    except Exception as error:
        failed[name] = f"{type(error).__name__}: {error}"
print(json.dumps(failed))
"""


def test_every_module_imports_first() -> None:
    """Every shipped module imported as the first of the workspace's modules a
    process reaches: in fresh interpreters, since this one has imported everything
    by now, with the workspace's modules dropped before each. A cycle shows only
    when it is entered from the wrong side, and any module may be the one a process
    enters by: the daemon, a runtime venv, a test. Nothing may fail.

    Split across processes, since each module is imported on its own either way:
    serially this took about 25s.

    Sabotage: import `ph.seams.changes` at the top of `ph.seams.workspace_jj`, which
    `changes` imports `jj` from, and `ph.seams.workspace_jj` (and `ph.testing.jj`)
    fail here, where importing every module first to last and then last to first
    both pass: another module has always loaded `changes` by then.
    """
    names = sorted(name for name in parsed_modules() if not name.endswith("__main__"))
    tops = json.dumps(sorted(package.name for package in workspace_packages()))
    workers = min(8, os.cpu_count() or 1)
    probes = [
        subprocess.Popen(
            [sys.executable, "-c", PROBE, json.dumps(names[start::workers]), tops],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for start in range(workers)
    ]
    failed: dict[str, str] = {}
    for probe in probes:
        out, err = probe.communicate(timeout=300)
        assert probe.returncode == 0, err
        failed |= json.loads(out)
    assert failed == {}, failed
