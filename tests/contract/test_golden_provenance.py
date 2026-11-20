"""Every golden says what produced it: the licensed-derived goldens in the models directory and the tracked
random-weight goldens.

Each golden stamps ``provenance``, a JSON object ``{oracle, generator, asset_sha256, dtype, seed}``. Licensed-derived
goldens live in ``$BETTERHUMAN_MODELS_DIR/goldens/<family>/``, so their check runs in the release tier, and git holds
random-weight goldens only: those of ``tests/parity/smplx/random/<family>/``, computed from
``tests/support/random_weights/<family>_random.npz``. The
oracle is the version pinned in its tool folder, the generator and dtype are the family's dump script (which exists
in this repository) and oracle precision, the asset hash is 64 hex digits (for a random-weight golden, the hash of
its archive) and the seed an integer. A family directory without goldens fails too. Git holds no array file of any
format (``ARRAY_FILES``) but those goldens, the random-weight archives (each the bytes ``make_random_archive.py``
writes from its seed), the format fixtures and the shipped data.
"""

from __future__ import annotations

import hashlib
import json
import re
import runpy
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.support.goldens import golden_dir

REPO = Path(__file__).parents[2]
RANDOM_GOLDENS = REPO / "tests" / "parity" / "smplx" / "random"
RANDOM_ARCHIVES = REPO / "tests" / "support" / "random_weights"
MAKER = "tools/parity/smplx/make_random_archive.py"
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
#: Every ``.npz`` git may hold besides the random-weight goldens: the shipped data, the format fixtures and the
#: random-weight archives.
TRACKED_NPZ = {
    "src/better_human/io/assets/density_tables.npz",
    "src/better_human/io/assets/mhr2smplx_mapping.npz",
    "src/better_human/io/assets/smpl_segmentation.npz",
    *(
        f"tests/support/formats/{name}.npz"
        for name in ("density_tables_v1", "mhr_v1", "mhr_v2", "segmentation_v1", "soma_v1", "soma_v2")
    ),
    *(f"tests/support/random_weights/{family}_random.npz" for family in RANDOM),
}
#: Every file format an array or a set of weights can travel in.
ARRAY_FILES = ("*.npz", "*.npy", "*.pt", "*.pth", "*.ckpt", "*.pkl", "*.pickle", "*.joblib", "*.h5", "*.hdf5")
ARRAY_FILES += ("*.safetensors", "*.mat", "*.onnx", "*.bin", "*.tflite", "*.msgpack")
KEYS = {"oracle", "generator", "asset_sha256", "dtype", "seed"}
ASSET_SHA256 = re.compile(r"[0-9a-f]{64}")


def _pin(package: str) -> str:
    """``<package>==<version>`` as the package's pin file states it."""
    text = (REPO / PINS[package]).read_text()
    match = re.search(rf"^[\s\"'-]*{re.escape(package)}==?([0-9][^\s\"',]*)", text, re.MULTILINE)
    assert match, f"{PINS[package]} pins no {package} version"
    return f"{package}=={match.group(1)}"


def _listing_sha256(path: Path) -> str:
    """The ``asset_sha256`` of a one-file asset: sha256 of its ``sha256sum`` line."""
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


def test_git_holds_no_array_file_but_the_allowed_ones() -> None:
    """Every array file (``ARRAY_FILES``) git tracks, or would add (untracked and not ignored), is
    a random-weight golden or in ``TRACKED_NPZ``, so licensed weights or a golden computed from them, copied anywhere in
    the repository in any array format, fail."""
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", *ARRAY_FILES],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = {name for name in listing.split("\0") if name}
    goldens = {
        path.relative_to(REPO).as_posix() for family in RANDOM for path in (RANDOM_GOLDENS / family).glob("*.npz")
    }
    unexpected, missing = sorted(found - goldens - TRACKED_NPZ), sorted(TRACKED_NPZ - found)
    assert not unexpected, f"array files outside the allow-list (licensed data stays out of git): {unexpected}"
    assert not missing, f"TRACKED_NPZ lists files git does not hold: {missing}"


@pytest.mark.slow
@pytest.mark.parametrize("family", RANDOM)
def test_each_random_weight_archive_is_what_its_maker_writes(tmp_path: Path, family: str) -> None:
    """``make_random_archive.py`` regenerates each tracked archive from its seed, byte for byte."""
    maker = runpy.run_path(str(REPO / MAKER), run_name="make_random_archive")
    written = tmp_path / f"{family}_random.npz"
    maker["write_archive"](written, maker["make_arrays"](family))
    tracked = RANDOM_ARCHIVES / written.name
    assert hashlib.sha256(written.read_bytes()).hexdigest() == hashlib.sha256(tracked.read_bytes()).hexdigest(), (
        f"{tracked.relative_to(REPO)} is not what {MAKER} writes from seed {maker['SEED']}; regenerate it and its goldens"
    )
