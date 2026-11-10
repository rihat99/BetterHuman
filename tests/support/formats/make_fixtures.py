"""Write the frozen fixtures of ``tests/contract/test_formats.py`` and their ``SHA256SUMS`` listing.

Usage, from the repository root (the fixtures are frozen: regenerate only to add a format or a version, into a
scratch directory, and copy the new files over)::

    uv run python tests/support/formats/make_fixtures.py OUT_DIR

Every archive field is a distinct constant (its position in ``WRITTEN``, the field order the fixtures were frozen with)
at the schema's shape: the axes a schema binds keep its sizes (SOMA's 18056 vertices), the free ones are small
(``FREE``). Numbering by the frozen order, not the live schema's, keeps a rerun byte-identical to ``SHA256SUMS`` when a
schema drops a field the converters still write (``unit`` and ``forward_axis``, BH record 0008). The v1 archives are
written as the converters wrote them before decision record 0012: no ``format`` and no ``provenance``; MHR's
``coordinate_axis``, SOMA's ``lod`` and its provenance JSON ``source``. The frame sets hold the same three frames in
both versions; the studio's segmentation and pose (version 1 only) are two small hand-written values.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from better_human.io import write_frame_set
from better_human.studio.persistence import Pose, write_pose, write_segmentation
from better_human.io.schema import (
    DENSITY_TABLES_SCHEMA,
    MHR_SCHEMA,
    SEGMENTATION_SCHEMA,
    SOMA_SCHEMA,
    VERTEX_COORDINATES,
    ModelSchema,
)

#: Sizes of the axes a schema leaves free.
FREE = {"V": 4, "F": 2, "K": 2, "C": 4, "H": 3, "N": 5, "T": 2, "P+1": 3}
#: Fields written per version rather than as the schema's constants.
HEADER = ("format", "format_version", "provenance")
#: ``short name -> (schema, fields only v1 carries)``.
ARCHIVES = {
    "mhr": (MHR_SCHEMA, {"coordinate_axis": np.asarray("+x")}),
    "soma": (SOMA_SCHEMA, {"lod": np.asarray("mid"), "source": np.asarray('{"package": "py-soma-x"}')}),
}
#: ``short name -> schema`` of the archives with one version, written with their header and constants only.
FIRST_VERSION = {"segmentation": SEGMENTATION_SCHEMA, "density_tables": DENSITY_TABLES_SCHEMA}
#: ``short name -> field order`` the fixtures were written in: the schemas of 2026-11-06, header first.
WRITTEN = {
    "mhr": (
        *HEADER,
        *(
            "lod joint_names joint_parents joint_pre_rotations joint_translation_offsets segmentation_group_names "
            "segmentation_joint_group_indices model_parameter_names model_parameter_channel_types "
            "parameter_transform_rows parameter_transform_columns parameter_transform_values "
            "model_parameter_lower_limits model_parameter_upper_limits template_vertices faces skin_indices "
            "skin_weights identity_basis expression_basis up_axis forward_axis bind_matrices inverse_bind_matrices "
            "corrective_first_layer_indices corrective_first_layer_weights corrective_first_layer_mask "
            "corrective_output_weights"
        ).split(),
    ),
    "soma": (
        *HEADER,
        *(
            "unit up_axis forward_axis shape_mean shape_basis bind_vertices faces joint_names joint_parents "
            "bind_pose_world t_pose_world skin_indices skin_weights public_joint_ids public_parents "
            "public_skin_weights joint_regressor twist_joint_ids twist_segment twist_fraction twist_rotation_matrix "
            "segment_start segment_end segment_parent segment_reverse corrective_bind corrective_first_layer "
            "corrective_output_indices corrective_output_values corrective_output_shape corrective_tanh"
        ).split(),
    ),
    "segmentation": (*HEADER, *"num_vertices faces part_offsets".split()),
    "density_tables": (*HEADER, *"table_names categories densities".split()),
}
#: ``short name -> fields with real values``: the ones the loaders check, and the two the converters still write
#: that the schemas no longer list.
CHECKED = {
    "mhr": {"lod": np.asarray(1, dtype=np.int64), "up_axis": np.asarray("+y"), "forward_axis": np.asarray("+z")},
    "soma": {"unit": np.asarray("m"), "up_axis": np.asarray("+y"), "forward_axis": np.asarray("+z")},
}


def _fields(short: str, schema: ModelSchema) -> dict[str, np.ndarray]:
    """Every field of ``WRITTEN[short]`` but the header, filled with its 1-based position there (``CHECKED``'s real)."""
    sizes = {**FREE, **schema.dimensions}
    checked = CHECKED.get(short, {})
    fields = {}
    for index, name in enumerate(WRITTEN[short], start=1):
        if name in HEADER:
            continue
        if name in checked:
            fields[name] = checked[name]
            continue
        spec = schema.fields[name]
        shape = tuple(
            axis if isinstance(axis, int) else sizes["V"] * 3 if axis == VERTEX_COORDINATES else sizes[axis]
            for axis in spec.shape
        )
        dtype = np.dtype({"str": "<U4"}.get(spec.dtype, spec.dtype))
        if dtype.kind == "U":
            fields[name] = np.full(shape, f"f{index}", dtype=dtype)
        elif dtype.kind == "b":
            fields[name] = np.full(shape, index % 2 == 1)
        else:
            fields[name] = np.full(shape, index, dtype=dtype)
    return fields


def main(out: Path) -> None:
    """Write ``{mhr,soma}_v{1,2}.npz``, ``{segmentation,density_tables}_v1.npz``, ``frame_set_v{0,1}.json``,
    ``studio_{segmentation,pose}_v1.json`` and ``SHA256SUMS`` into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    for short, (schema, v1_only) in ARCHIVES.items():
        body = _fields(short, schema)
        np.savez_compressed(out / f"{short}_v1.npz", format_version=np.asarray(1, dtype=np.int64), **v1_only, **body)
        header = {
            "format": np.asarray(schema.format),
            "format_version": np.asarray(2, dtype=np.int64),
            "provenance": np.asarray('{"converter": "tests/support/formats fixture"}'),
        }
        np.savez_compressed(out / f"{short}_v2.npz", **header, **body)
    for short, schema in FIRST_VERSION.items():
        header = {
            "format": np.asarray(schema.format),
            "format_version": np.asarray(1, dtype=np.int64),
            "provenance": np.asarray('{"converter": "tests/support/formats fixture"}'),
        }
        np.savez_compressed(out / f"{short}_v1.npz", **header, **_fields(short, schema))
    v0 = {"legacy": 331, "default": {"vertex": 100, "joint": None}, "named": {"vertex": 200, "joint": "left_wrist"}}
    (out / "frame_set_v0.json").write_text(json.dumps(v0, indent=2) + "\n")
    write_frame_set(
        out / "frame_set_v1.json", {"legacy": (331, None), "default": (100, None), "named": (200, "left_wrist")}
    )
    write_segmentation(out / "studio_segmentation_v1.json", {"head": [0, 1, 2], "torso": [3, 4]})
    write_pose(out / "studio_pose_v1.json", Pose(joint_names=("root", "spine"), q=(0.0, 0.5, -0.25)))
    fixtures = sorted(path for path in out.iterdir() if path.suffix in {".npz", ".json"})
    listing = "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in fixtures)
    (out / "SHA256SUMS").write_text(listing)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
