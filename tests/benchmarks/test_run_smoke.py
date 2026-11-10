"""Smoke tests of ``benchmarks/run.py``: wall time with its context, and the rows that read licensed weights.

The counters and their thresholds are checked by ``tests/contract/test_compile_counters.py``. Without weights the
runner times its counter rows on the random-weight archives; ``--licensed`` adds the sweep, whose rows at batch 1024
are re-timed with fixed allocator thresholds, and the load rows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(out: Path, *arguments: str) -> dict[str, Any]:
    """The runner's JSON after ``--timing-only`` with ``arguments``, in a child with the internal checks off."""
    command = [sys.executable, str(REPO_ROOT / "benchmarks" / "run.py"), "--timing-only", "--repeats", "3"]
    command += [*arguments, "--out", str(out)]
    env = {**os.environ, "BETTER_ROBOT_INTERNAL_CHECKS": "0"}
    for name in ("MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_"):  # set, they would stop the fixed-allocator rerun
        env.pop(name, None)
    completed = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600, check=False
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    return json.loads(out.read_text())


def test_runner_times_a_counter_row_with_its_context(tmp_path: Path) -> None:
    result = _run(tmp_path / "run.json", "--rows", "fk_vertices/smplx/B1")
    meta, (row,) = result["meta"], result["rows"]
    assert row["row"] == "fk_vertices/smplx/B1"
    assert {"median_ms", "min_ms", "page_faults", "loadavg"} <= row.keys()
    assert {"commit", "dirty", "command", "torch", "device_name", "env", "loadavg_start", "loadavg_end"} <= meta.keys()


@pytest.mark.licensed
def test_licensed_rows_time_a_large_batch_with_both_allocators_and_a_load(tmp_path: Path) -> None:
    sweep_row, load_row = "SMPL/fk/B1024/default", "load/soma"
    result = _run(tmp_path / "run.json", "--licensed", "--batches", "1024", "--rows", f"{sweep_row},{load_row}")
    rows = {row["row"]: row for row in result["rows"]}
    assert rows.keys() == {sweep_row, load_row}
    assert {"median_ms", "page_faults", "fixed_malloc_median_ms", "fixed_malloc_page_faults"} <= rows[sweep_row].keys()
    assert {"median_ms", "raw_read_ms", "ratio"} <= rows[load_row].keys()
