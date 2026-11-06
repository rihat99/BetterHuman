"""Every licensed-derived golden in the models directory says what produced it.

Decision records 0014 (each golden stamps ``provenance``, a JSON object ``{oracle, generator, asset_sha256, dtype,
seed}``) and 0015 (those goldens live in ``$BETTERHUMAN_MODELS_DIR/goldens/<family>/``, so this check runs in the
release tier). The oracle is a pinned ``<package>==<version>``, the generator a dump script that exists in this
repository, the asset hash 64 hex digits or ``unknown``, the dtype a float dtype and the seed an integer. A family
directory without goldens fails too.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from tests.support.goldens import golden_dir

REPO = Path(__file__).parents[2]
FAMILIES = ("smpl_neutral", "smplx_neutral", *(f"mhr_lod{lod}" for lod in range(7)), "soma_mid")
KEYS = {"oracle", "generator", "asset_sha256", "dtype", "seed"}
ORACLE = re.compile(r"[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.+!-]*")
ASSET_SHA256 = re.compile(r"[0-9a-f]{64}|unknown")
DTYPES = {"float16", "bfloat16", "float32", "float64"}


@pytest.mark.licensed
@pytest.mark.parametrize("family", FAMILIES)
def test_licensed_goldens_stamp_provenance(family: str) -> None:
    directory = golden_dir(family)
    files = sorted(directory.glob("*.npz"))
    assert files, f"no goldens in {directory}"
    for path in files:
        with np.load(path, allow_pickle=False) as golden:
            assert "provenance" in golden.files, f"{path} has no provenance"
            provenance = json.loads(str(golden["provenance"]))
        assert set(provenance) == KEYS, f"{path}: provenance keys {sorted(provenance)}, expected {sorted(KEYS)}"
        assert ORACLE.fullmatch(provenance["oracle"]), f"{path}: oracle {provenance['oracle']!r} is not pinned"
        assert (REPO / provenance["generator"]).is_file(), f"{path}: generator {provenance['generator']} is missing"
        assert ASSET_SHA256.fullmatch(provenance["asset_sha256"]), (
            f"{path}: asset_sha256 {provenance['asset_sha256']!r}"
        )
        assert provenance["dtype"] in DTYPES, f"{path}: dtype {provenance['dtype']!r}"
        assert type(provenance["seed"]) is int, f"{path}: seed {provenance['seed']!r} is not an integer"
