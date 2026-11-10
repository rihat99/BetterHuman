"""The random-weight archives (decision record 0016), each read once per test session.

The SMPL-X archive inflates to 166 MB of arrays, and on a host with transparent huge pages one read of it can stall
for about 10 s, so every test that loads an archive reads it through :func:`random_weight_arrays` or builds its spec
through :func:`random_weight_spec`: the family's full loader runs for every option set, on arrays read once per path.
Callers never write to the cached arrays, which the specs built from them share.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch

import better_human as bh
from better_human.families import smpl_family
from better_human.io.load import load_model_npz
from better_human.io.schema import SMPL_SCHEMA, SMPLH_SCHEMA, SMPLX_SCHEMA, ModelSchema

ARCHIVES = Path(__file__).parent / "random_weights"
SCHEMAS = {"smpl": SMPL_SCHEMA, "smplh": SMPLH_SCHEMA, "smplx": SMPLX_SCHEMA}
_READS: dict[Path, dict[str, torch.Tensor | np.ndarray]] = {}
#: Two contact frames of the random SMPL-X archive: one whose parent comes from the part ids, one named.
SMPLX_FRAMES = {"left_heel": (8846, None), "right_heel": (8635, "right_ankle")}


def _read_once(path: Path, schema: ModelSchema) -> dict[str, torch.Tensor | np.ndarray]:
    """``io.load.load_model_npz`` that reads each path once per session."""
    if path not in _READS:
        _READS[path] = load_model_npz(path, schema)
    return dict(_READS[path])


def random_weight_arrays(family: str) -> dict[str, torch.Tensor | np.ndarray]:
    """The arrays of ``<family>_random.npz`` as the family's loader reads them; read once per session."""
    return _read_once(ARCHIVES / f"{family}_random.npz", SCHEMAS[family])


def random_weight_spec(family: str, **options: Any) -> bh.FamilySpec:
    """``bh.families.<family>(<family>_random.npz, **options)``: the full loader on the arrays read once per session."""
    with mock.patch.object(smpl_family, "load_model_npz", _read_once):
        return getattr(bh.families, family)(ARCHIVES / f"{family}_random.npz", **options)


@cache
def random_weight_smplx() -> bh.HumanModel:
    """The random SMPL-X archive with Dempster part inertias and :data:`SMPLX_FRAMES`, built once per session."""
    return bh.HumanModel.from_spec(random_weight_spec("smplx"), contact_frames=SMPLX_FRAMES)
