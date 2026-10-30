"""Golden parity against the pinned float32 SOMA-X v0.3.4 oracle (``tools/parity/soma``, mid LOD)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

from better_human import Body, HumanState, families
from better_human.families.soma import SOMAClassic, soma_from_classic

ROOT = Path(__file__).parents[2]
MODEL_PATH = ROOT / "models" / "SOMA" / "soma_mid.npz"
GOLDEN_FILES = tuple(sorted((Path(__file__).parent / "goldens" / "soma_mid").glob("*.npz")))
EXPECTED_CASES = {
    "big_shape",
    "correctives_off",
    *(f"identity_{index:02d}" for index in range(3)),
    *(f"pose_{index:02d}" for index in range(5)),
    "reflect_00",
    "twist_00",
    "twist_01",
    "zero",
}
assert {path.stem for path in GOLDEN_FILES} == EXPECTED_CASES, "The committed SOMA parity grid is missing or unexpected"
KEYS = ("identity_coeffs", "poses", "transl", "vertices", "skin_transforms")
LABELS = ("vertex", "joint position", "rotation entry")

# Vertices [m], joint positions [m], rotation entries, float32 and float64: 2x the worst error over the 14 goldens and
# their identity batch, rounded up. A float32 joint regression alone rounds by up to 5.2e-6 m on these identities, so
# a correct float64 or batched float32 evaluation sits that far from the float32 oracle; no gate goes below that.
# Single reflect_00, a 5-sigma identity, sets its own vertex gate and not the general one: its float32 joint regression
# rounds further (5.4e-6 m; 3.0e-6 m with that regression in float64), as upstream's does (7.5e-6 m from its float64).
PARITY_ATOLS = (6.5e-6, 9e-6, 3.5e-6)
PARITY_ATOLS_BY_CASE = {"reflect_00": (1.1e-5, 9e-6, 3.5e-6)}
PARITY_CASES = tuple((dtype, path) for dtype in (torch.float32, torch.float64) for path in GOLDEN_FILES)


@lru_cache(maxsize=2)
def _body(dtype: torch.dtype, correctives: bool) -> Body:
    if not MODEL_PATH.is_file():
        pytest.skip(f"converted SOMA fixture is absent: {MODEL_PATH}")
    return Body(families.soma(MODEL_PATH, correctives=correctives, dtype=dtype), mass=False)


def _golden(path: Path) -> tuple[bool, dict[str, torch.Tensor]]:
    """``(apply_correctives, float32 tensors of KEYS)`` of one golden case."""
    with np.load(path, allow_pickle=False) as golden:
        assert int(golden["format_version"]) == 2
        assert str(golden["case_name"]) == path.stem
        return bool(golden["apply_correctives"]), {name: torch.from_numpy(golden[name]) for name in KEYS}


def _errors(state: HumanState, expected: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Max vertex, joint position and rotation entry errors ``(...)`` per case, over all 110 skin joints.

    ``skin_transforms`` holds upstream's public ``joints`` and ``transforms`` bit for bit, plus the twist helpers.
    """
    skin = expected["skin_transforms"][..., :3, :]
    return (
        (state.vertices - expected["vertices"]).abs().amax((-2, -1)),
        (state.joints - skin[..., 3]).abs().amax((-2, -1)),
        (state.joint_transforms[..., :3] - skin[..., :3]).abs().amax((-3, -2, -1)),
    )


@pytest.mark.parametrize(
    ("dtype", "golden_path"),
    PARITY_CASES,
    ids=[f"{str(dtype).removeprefix('torch.')}-{path.stem}" for dtype, path in PARITY_CASES],
)
def test_soma_parity(dtype: torch.dtype, golden_path: Path) -> None:
    """Vertices and the world transforms of every skin joint against one upstream case."""
    correctives, golden = _golden(golden_path)
    expected = {name: value.to(dtype) for name, value in golden.items()}
    with torch.inference_mode():
        classic = SOMAClassic(expected["identity_coeffs"], expected["poses"], expected["transl"])
        identity, q, stream = soma_from_classic(_body(dtype, correctives), classic)
        errors = _errors(identity.forward(q, stream), expected)
    atols = PARITY_ATOLS_BY_CASE.get(golden_path.stem, PARITY_ATOLS)
    for label, error, atol in zip(LABELS, errors, atols, strict=True):
        assert error <= atol, f"{dtype} {golden_path.stem}: {label} max error {error.item():.3e}"


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=("float32", "float64"))
def test_soma_parity_identity_batch(dtype: torch.dtype) -> None:
    """Every correctives-on golden in one identity batch, one pose each: batched sums round in another order."""
    goldens = {path.stem: _golden(path) for path in GOLDEN_FILES}
    names = [name for name, (correctives, _) in goldens.items() if correctives]
    expected = {key: torch.stack([goldens[name][1][key] for name in names]).to(dtype) for key in KEYS}
    with torch.inference_mode():
        classic = SOMAClassic(expected["identity_coeffs"], expected["poses"], expected["transl"])
        identity, q, stream = soma_from_classic(_body(dtype, True), classic)
        errors = _errors(identity.forward(q, stream), expected)
    assert q.shape == (len(names), 311)
    for label, error, atol in zip(LABELS, errors, PARITY_ATOLS, strict=True):
        worst = int(error.argmax())
        assert error[worst] <= atol, f"{dtype} {names[worst]} in a batch: {label} max error {error[worst].item():.3e}"
