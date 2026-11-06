"""Every licensed-derived golden in the models directory says what produced it.

Decision records 0014 (each golden stamps ``provenance``, a JSON object ``{oracle, generator, asset_sha256, dtype,
seed}``) and 0015 (those goldens live in ``$BETTERHUMAN_MODELS_DIR/goldens/<family>/``, so this check runs in the
release tier). The oracle is a pinned ``<package>==<version>``, the generator and dtype are the family's dump script
(which exists in this repository) and oracle precision in ``FAMILIES``, the asset hash is 64 hex digits or
``unknown`` and the seed an integer. A family directory without goldens fails too.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from tests.support.goldens import golden_dir

REPO = Path(__file__).parents[2]
#: family -> (generator, dtype): the dump script that writes the family's goldens and the oracle's precision.
FAMILIES: dict[str, tuple[str, str]] = {
    "smpl_neutral": ("tools/parity/smplx/dump_smpl.py", "float64"),
    "smplx_neutral": ("tools/parity/smplx/dump_smplx.py", "float64"),
    **{f"mhr_lod{lod}": ("tools/parity/mhr/dump_mhr.py", "float32") for lod in range(7)},
    "soma_mid": ("tools/parity/soma/dump_soma.py", "float32"),
}
KEYS = {"oracle", "generator", "asset_sha256", "dtype", "seed"}
ORACLE = re.compile(r"[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.+!-]*")
ASSET_SHA256 = re.compile(r"[0-9a-f]{64}|unknown")


@pytest.mark.licensed
@pytest.mark.parametrize("family", FAMILIES)
def test_licensed_goldens_stamp_provenance(family: str) -> None:
    generator, dtype = FAMILIES[family]
    assert (REPO / generator).is_file(), f"{family}: generator {generator} is missing"
    directory = golden_dir(family)
    files = sorted(directory.glob("*.npz"))
    assert files, f"no goldens in {directory}"
    for path in files:
        with np.load(path, allow_pickle=False) as golden:
            assert "provenance" in golden.files, f"{path} has no provenance"
            provenance = json.loads(str(golden["provenance"]))
        assert set(provenance) == KEYS, f"{path}: provenance keys {sorted(provenance)}, expected {sorted(KEYS)}"
        assert ORACLE.fullmatch(provenance["oracle"]), f"{path}: oracle {provenance['oracle']!r} is not pinned"
        assert provenance["generator"] == generator, (
            f"{path}: generator {provenance['generator']!r}, expected {generator!r}"
        )
        assert ASSET_SHA256.fullmatch(provenance["asset_sha256"]), (
            f"{path}: asset_sha256 {provenance['asset_sha256']!r}"
        )
        assert provenance["dtype"] == dtype, f"{path}: dtype {provenance['dtype']!r}, expected {dtype!r}"
        assert type(provenance["seed"]) is int, f"{path}: seed {provenance['seed']!r} is not an integer"
