"""The CPU counters of the hot entry points never grow (BetterRobot ``C-PERF-3`` and decision record 0011).

``benchmarks/run.py --counters-only`` measures on CPU, for every counter row its docstring lists, the graph breaks,
recompiles, top-level operators and host-syncing operators on the random-weight archives, and compares them with
``benchmarks/thresholds/torch-<version>.json``. It exits 1 when a counter is above its threshold, a row has no
threshold, or the installed torch has no thresholds file. It runs in a child process with BetterRobot's internal
checks off: they add host syncs and operators, and the switch is read once, at import. Marked ``slow`` (about a
minute).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_cpu_counters_stay_within_their_thresholds(tmp_path: Path) -> None:
    """A grown counter fails: remove the new graph break, recompile, operator or host sync it names."""
    command = [sys.executable, str(REPO / "benchmarks" / "run.py"), "--counters-only", "--device", "cpu"]
    command += ["--out", str(tmp_path / "counters.json")]
    env = {**os.environ, "BETTER_ROBOT_INTERNAL_CHECKS": "0"}
    completed = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, timeout=900, check=False)
    assert completed.returncode == 0, (
        "a CPU counter is above its threshold (BetterRobot C-PERF-3): remove what added it; a change that needs it, or "
        "that lowers one, rewrites benchmarks/thresholds/ on a clean commit in a commit of its own with a Why: line\n"
        + completed.stdout[-4000:]
        + completed.stderr[-2000:]
    )
