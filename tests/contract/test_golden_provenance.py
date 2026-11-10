"""Every golden says what produced it: the licensed-derived goldens in the models directory and the tracked
random-weight goldens.

Decision records 0014 (each golden stamps ``provenance``, a JSON object ``{oracle, generator, asset_sha256, dtype,
seed}``), 0015 (licensed-derived goldens live in ``$BETTERHUMAN_MODELS_DIR/goldens/<family>/``, so their check runs
in the release tier, and git holds random-weight goldens only) and 0016 (the random-weight goldens of
``tests/parity/smplx/random/<family>/``, computed from ``tests/support/random_weights/<family>_random.npz``). The
oracle is the version pinned in its tool folder, the generator and dtype are the family's dump script (which exists
in this repository) and oracle precision, the asset hash is 64 hex digits (for a random-weight golden, the hash of
its archive) and the seed an integer. A family directory without goldens fails too.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from tests.support.goldens import golden_dir

REPO = Path(__file__).parents[2]
RANDOM_GOLDENS = REPO / "tests" / "parity" / "smplx" / "random"
RANDOM_ARCHIVES = REPO / "tests" / "support" / "random_weights"
#: oracle package -> the file in its tool folder that pins its version (``==`` in pip files, ``=`` in conda files).
PINS = {
    "smplx": "tools/parity/smplx/pyproject.toml",
    "pymomentum": "tools/parity/mhr/environment.yml",
    "py-soma-x": "tools/parity/soma/requirements.txt",
}
#: family -> (oracle package, generator, dtype): the dump script that writes the family's goldens and its precision.
LICENSED: dict[str, tuple[str, str, str]] = {
    "smpl_neutral": ("smplx", "tools/parity/smplx/dump_smpl.py", "float64"),
    "smplx_neutral": ("smplx", "tools/parity/smplx/dump_smplx.py", "float64"),
    **{f"mhr_lod{lod}": ("pymomentum", "tools/parity/mhr/dump_mhr.py", "float32") for lod in range(7)},
    "soma_mid": ("py-soma-x", "tools/parity/soma/dump_soma.py", "float32"),
}
RANDOM: dict[str, tuple[str, str, str]] = {
    "smpl": ("smplx", "tools/parity/smplx/dump_smpl.py", "float64"),
    "smplh": ("smplx", "tools/parity/smplx/dump_smplh.py", "float64"),
    "smplx": ("smplx", "tools/parity/smplx/dump_smplx.py", "float64"),
}
KEYS = {"oracle", "generator", "asset_sha256", "dtype", "seed"}
ASSET_SHA256 = re.compile(r"[0-9a-f]{64}")


def _pin(package: str) -> str:
    """``<package>==<version>`` as the package's pin file states it."""
    text = (REPO / PINS[package]).read_text()
    match = re.search(rf"^[\s\"'-]*{re.escape(package)}==?([0-9][^\s\"',]*)", text, re.MULTILINE)
    assert match, f"{PINS[package]} pins no {package} version"
    return f"{package}=={match.group(1)}"


def _listing_sha256(path: Path) -> str:
    """The ``asset_sha256`` of a one-file asset: sha256 of its ``sha256sum`` line (decision record 0014)."""
    return hashlib.sha256(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n".encode()).hexdigest()


def _checked_provenance(path: Path, package: str, generator: str, dtype: str) -> dict[str, object]:
    """The provenance of the golden ``path``, asserted against the family's oracle pin, generator and dtype."""
    with np.load(path, allow_pickle=False) as golden:
        assert "provenance" in golden.files, f"{path} has no provenance"
        provenance = json.loads(str(golden["provenance"]))
    assert set(provenance) == KEYS, f"{path}: provenance keys {sorted(provenance)}, expected {sorted(KEYS)}"
    assert provenance["oracle"] == _pin(package), f"{path}: oracle {provenance['oracle']!r}, pinned {_pin(package)!r}"
    assert provenance["generator"] == generator, (
        f"{path}: generator {provenance['generator']!r}, expected {generator!r}"
    )
    assert ASSET_SHA256.fullmatch(provenance["asset_sha256"]), f"{path}: asset_sha256 {provenance['asset_sha256']!r}"
    assert provenance["dtype"] == dtype, f"{path}: dtype {provenance['dtype']!r}, expected {dtype!r}"
    assert type(provenance["seed"]) is int, f"{path}: seed {provenance['seed']!r} is not an integer"
    return provenance


@pytest.mark.licensed
@pytest.mark.parametrize("family", LICENSED)
def test_licensed_goldens_stamp_provenance(family: str) -> None:
    package, generator, dtype = LICENSED[family]
    assert (REPO / generator).is_file(), f"{family}: generator {generator} is missing"
    directory = golden_dir(family)
    files = sorted(directory.glob("*.npz"))
    assert files, f"no goldens in {directory}"
    for path in files:
        _checked_provenance(path, package, generator, dtype)


@pytest.mark.parametrize("family", RANDOM)
def test_random_weight_goldens_name_their_archive(family: str) -> None:
    package, generator, dtype = RANDOM[family]
    assert (REPO / generator).is_file(), f"{family}: generator {generator} is missing"
    archive = RANDOM_ARCHIVES / f"{family}_random.npz"
    files = sorted((RANDOM_GOLDENS / family).glob("*.npz"))
    assert files, f"no goldens in {RANDOM_GOLDENS / family}"
    for path in files:
        provenance = _checked_provenance(path, package, generator, dtype)
        assert provenance["asset_sha256"] == _listing_sha256(archive), (
            f"{path}: asset_sha256 is not the hash of {archive.relative_to(REPO)}; regenerate it from that archive"
        )


def test_the_repository_holds_random_weight_goldens_only() -> None:
    """Decision record 0015: a golden in ``tests/parity`` is one of the random-weight families' goldens."""
    goldens = {path.relative_to(REPO) for path in (REPO / "tests" / "parity").rglob("*.npz")}
    random_weight = {path.relative_to(REPO) for family in RANDOM for path in (RANDOM_GOLDENS / family).glob("*.npz")}
    assert goldens == random_weight, f"goldens outside the random-weight families: {sorted(goldens - random_weight)}"
