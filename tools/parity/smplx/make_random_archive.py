"""Write a seeded random-weight SMPL, SMPL-H or SMPL-X archive in the upstream layout (decision record 0016).

The archive has the family's native vertex and joint counts and the fields and dtypes that BetterHuman's loader and
the ``smplx`` oracle read, but every array is drawn from one seeded generator: no number comes from licensed
weights. Values are small integers times a power of two, exact in float32 and float64, and the shape and pose blend
shapes are non-zero on 256 random vertices only, so each archive deflates to under 1 MiB (principle 9.3). The joint
regressor reads those vertices only, so every joint, the root included, moves with shape and expression and a gate on
these weights sees each joint's shape-dependent pivot. The kinematic tree is random: a joint's parent is an earlier
joint of its own group (body, face, left hand, right hand) or a body joint, so the hands and the face stay terminal
subtrees. The faces are a consistently wound strip over the vertex order, a band the mass code can close part by
part, so the archive also loads with a density; its masses are those of a random mesh, not of a body.

Every zip entry carries one fixed timestamp, so a seed always writes the same bytes with the same numpy and zlib::

    uv run --project tools/parity/smplx python tools/parity/smplx/make_random_archive.py smplx \\
        tests/support/random_weights/smplx_random.npz
"""

from __future__ import annotations

import argparse
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SEED = 20261110
#: Vertices that carry shape and pose blend shapes (every column of both bases is random on them) and that the joint
#: regressor reads.
BLENDED_VERTICES = 256


@dataclass(frozen=True)
class Layout:
    """What a family's upstream file holds: counts and the dtype of each kind of field."""

    num_vertices: int
    #: Joint count of each group in native order: the body first, then the face and hand groups.
    groups: tuple[int, ...]
    shape_columns: int
    floats: type[np.floating]  #: every float field but ``shapedirs``
    shapedirs: type[np.floating]
    kintree: type[np.integer]


FAMILIES = {
    "smpl": Layout(6890, (24,), 10, np.float64, np.float64, np.uint32),
    "smplh": Layout(6890, (22, 15, 15), 16, np.float64, np.float64, np.int64),  # AMASS SMPL+H: no hand PCA
    "smplx": Layout(10475, (22, 3, 15, 15), 400, np.float32, np.float64, np.int64),
}


def _steps(rng: np.random.Generator, shape: tuple[int, ...], *, step: float, largest: int) -> np.ndarray:
    """Integers in ``[-largest, largest]`` times ``step`` (a power of two)."""
    return rng.integers(-largest, largest + 1, size=shape) * step


def _partition(rng: np.random.Generator, count: int, total: int) -> np.ndarray:
    """``count`` positive multiples of ``1 / total`` that sum to 1 exactly."""
    cuts = np.sort(rng.choice(np.arange(1, total), size=count - 1, replace=False))
    return np.diff(np.concatenate(([0], cuts, [total]))) / total


def _convex_rows(
    rng: np.random.Generator, rows: int, columns: int, *, counts: tuple[int, ...], total: int
) -> np.ndarray:
    """``(rows, columns)``: each row is a partition of 1 (``_partition``) over a random set of columns, whose size is
    drawn from ``counts``."""
    matrix = np.zeros((rows, columns))
    for row in range(rows):
        count = int(rng.choice(counts))
        matrix[row, rng.choice(columns, size=count, replace=False)] = _partition(rng, count, total)
    return matrix


def _parents(rng: np.random.Generator, groups: tuple[int, ...]) -> list[int]:
    """A random tree: each joint's parent is an earlier joint of its own group or a body joint."""
    body = groups[0]
    parents = [-1, *(int(rng.integers(joint)) for joint in range(1, body))]
    start = body
    for size in groups[1:]:
        parents += [int(rng.choice([*range(body), *range(start, joint)])) for joint in range(start, start + size)]
        start += size
    return parents


def make_arrays(family: str, seed: int = SEED) -> dict[str, np.ndarray]:
    """The arrays of the random ``family`` archive drawn from ``seed``."""
    layout = FAMILIES[family]
    rng = np.random.default_rng(seed)
    num_vertices, num_joints = layout.num_vertices, sum(layout.groups)
    blended = np.sort(rng.choice(num_vertices, size=BLENDED_VERTICES, replace=False))

    def blend_shapes(columns: int, dtype: type[np.floating]) -> np.ndarray:
        basis = np.zeros((num_vertices, 3, columns), dtype=dtype)
        basis[blended] = _steps(rng, (blended.size, 3, columns), step=2**-8, largest=4)
        return basis

    parents = _parents(rng, layout.groups)
    regressor = np.zeros((num_joints, num_vertices))
    regressor[:, blended] = _convex_rows(rng, num_joints, blended.size, counts=(8,), total=64)
    root = np.iinfo(layout.kintree).max if np.issubdtype(layout.kintree, np.unsignedinteger) else -1
    strip = np.arange(num_vertices - 2)
    faces = np.stack((strip, strip + 1, strip + 2), axis=1)
    faces[1::2, :2] = faces[1::2, 1::-1]  # every other face reversed: neighbours cross their shared edge oppositely
    arrays = {
        "v_template": _steps(rng, (num_vertices, 3), step=2**-8, largest=128).astype(layout.floats),
        "shapedirs": blend_shapes(layout.shape_columns, layout.shapedirs),
        "posedirs": blend_shapes((num_joints - 1) * 9, layout.floats),
        "J_regressor": regressor.astype(layout.floats),
        "weights": _convex_rows(rng, num_vertices, num_joints, counts=(1, 2, 3, 4), total=16).astype(layout.floats),
        "kintree_table": np.array([[root, *parents[1:]], range(num_joints)], dtype=layout.kintree),
        "f": faces.astype(np.uint32),
    }
    if family == "smplx":
        for side in "lr":
            arrays[f"hands_components{side}"] = _steps(rng, (45, 45), step=2**-4, largest=8).astype(np.float32)
            arrays[f"hands_mean{side}"] = _steps(rng, (45,), step=2**-4, largest=4).astype(np.float32)
        arrays["lmk_faces_idx"] = rng.integers(0, len(arrays["f"]), size=51)
        arrays["lmk_bary_coords"] = np.stack([_partition(rng, 3, 8) for _ in range(51)])
    return arrays


def write_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write ``arrays`` as an ``.npz`` whose bytes depend on the arrays only (fixed entry timestamps)."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            archive.writestr(entry, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> None:
    """Write the random archive of the family named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("family", choices=sorted(FAMILIES))
    parser.add_argument("output", type=Path, help="The .npz file to write")
    args = parser.parse_args()
    write_archive(args.output, make_arrays(args.family))


if __name__ == "__main__":
    main()
