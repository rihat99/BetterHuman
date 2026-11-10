"""Every example runs headless with ``--check``: it builds its scene on BetterRobot's stand-in server
(``better_robot.testing.FakeServer``), moves something once, checks that the drawn arrays moved as asked (03: the
wrist reaches the target within 1e-4 m), prints one line and exits 0; a check that fails exits 1.

Examples 01-03 read the random-weight archives (record 0016) through ``--model-path``, so they need no licensed
weights; example 04 (MHR) reads them from ``$BETTERHUMAN_MODELS_DIR``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RANDOM_WEIGHTS = ROOT / "tests" / "support" / "random_weights"
EXAMPLE_TIMEOUT_S = 300

RUNS = [
    pytest.param("01_smpl_shape_pose.py", RANDOM_WEIGHTS / "smpl_random.npz", "scene with 2 layers", id="01"),
    pytest.param("02_smplx_partial.py", RANDOM_WEIGHTS / "smplx_random.npz", "90 frames, 5 layers", id="02"),
    pytest.param("03_ik_reach.py", RANDOM_WEIGHTS / "smpl_random.npz", "target moved", id="03"),
    pytest.param("04_mhr_shape_pose.py", None, "MHR nq=132", id="04", marks=pytest.mark.licensed),
]


@pytest.mark.slow
@pytest.mark.parametrize(("example", "weights", "printed"), RUNS)
def test_example_check_runs_headless(example: str, weights: Path | None, printed: str) -> None:
    command = [sys.executable, str(ROOT / "examples" / example), "--check"]
    if weights is not None:
        command += ["--model-path", str(weights)]
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment, timeout=EXAMPLE_TIMEOUT_S, check=False
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert printed in completed.stdout, completed.stdout
