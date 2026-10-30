"""Convert SOMA-X release assets into one self-contained BetterHuman NPZ archive (mid LOD, metres)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.sparse import csc_matrix
from soma import SOMALayer
from soma.io import load_lod_rigs_from_usd
from soma.units import Unit

DEFAULT_ASSETS_DIR = Path(
    "/tmp/claude-1003/-home-rikhat-akizhanov-better/364c020a-cd00-4ea3-ab3a-688eb158660a/scratchpad/soma_assets"
)
VERTEX_COUNT = 18056
JOINT_COUNT = 110
PUBLIC_JOINT_COUNT = 78
TWIST_COUNT = 32
SEGMENT_COUNT = 8
SHAPE_COUNT = 128
CORRECTIVES_PER_JOINT = 24
CENTIMETERS_TO_METERS = np.float32(0.01)
TWIST_MODE = "aligned_x_swing_twist"
ASSET_FILES = (
    "SOMA_neutral.npz",
    "SOMA_template_rig.usda",
    "SOMA_procedural_transforms.json",
    "correctives_model.pt",
)
MANIFEST_FILE = "SOMA-X-HF-MANIFEST.json"

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR, help="Directory with SOMA-X assets")
    parser.add_argument("--output-dir", type=Path, default=Path("models/SOMA"), help="Directory for soma_mid.npz")
    return parser.parse_args()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_metadata(assets_dir: Path) -> str:
    """Return a JSON string with the upstream version, asset tag and checked file hashes."""
    hashes = {name: _sha256(assets_dir / name) for name in ASSET_FILES}
    manifest_path = assets_dir / MANIFEST_FILE
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
        mismatched = [name for name, value in hashes.items() if expected.get(name) != value]
        if mismatched:
            raise ValueError(f"Asset hashes differ from {MANIFEST_FILE}: {mismatched}")
    return json.dumps(
        {
            "package": "py-soma-x",
            "package_version": importlib.metadata.version("py-soma-x"),
            "asset_release_tag": manifest.get("release_tag", "unknown"),
            "asset_commit": manifest.get("public_commit", "unknown"),
            "sha256": hashes,
        },
        sort_keys=True,
    )


def _transforms_to_meters(matrices: np.ndarray) -> np.ndarray:
    output = np.asarray(matrices, dtype=np.float32).copy()
    output[..., :3, 3] *= CENTIMETERS_TO_METERS
    return output


def _compact_skin(dense_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return lossless top-K ``(V, K)`` indices and weights; K is the largest row support."""
    support = int((dense_weights > 0).sum(axis=1).max())
    order = np.argsort(-dense_weights, axis=1, kind="stable")[:, :support]
    weights = np.take_along_axis(dense_weights, order, axis=1).astype(np.float32)
    order[weights == 0] = 0
    if not np.array_equal((dense_weights > 0).sum(axis=1), (weights > 0).sum(axis=1)):
        raise ValueError("Top-K skin compaction dropped nonzero weights")
    return order.astype(np.int64), weights


def _shape_data(neutral: Any) -> dict[str, np.ndarray]:
    """PCA mean and a basis pre-scaled by sqrt(eigenvalue), so a coefficient equals upstream identity_coeffs."""
    mean = np.asarray(neutral["mean"], dtype=np.float32).reshape(VERTEX_COUNT, 3)
    shapedirs = np.asarray(neutral["shapedirs"], dtype=np.float32).reshape(SHAPE_COUNT, VERTEX_COUNT, 3)
    scale = np.sqrt(np.asarray(neutral["eigenvalues"], dtype=np.float32))
    basis = np.transpose(shapedirs * scale[:, None, None], (1, 2, 0))
    return {
        "shape_mean": mean * CENTIMETERS_TO_METERS,
        "shape_basis": np.ascontiguousarray(basis * CENTIMETERS_TO_METERS, dtype=np.float32),
    }


def _check_faces(faces: np.ndarray, face_vert_indices: np.ndarray, face_vert_counts: np.ndarray) -> None:
    """Assert triangles 2i, 2i+1 split USD quad i with its winding (the diagonal may differ from a fan)."""
    if np.any(face_vert_counts != 4) or len(faces) != 2 * len(face_vert_counts):
        raise ValueError("Expected an all-quad USD skin mesh with two triangles per quad")
    quads = face_vert_indices.reshape(-1, 4)
    pairs = faces.reshape(-1, 2, 3)
    matches = pairs[..., None] == quads[:, None, None, :]
    if not np.all(matches.sum(axis=-1) == 1) or not np.all(matches.any(axis=(1, 2))):
        raise ValueError("SOMA_neutral.npz triangles do not cover the USD quads")
    positions = matches.argmax(axis=-1)
    steps = (np.roll(positions, -1, axis=-1) - positions) % 4
    if not np.all((steps > 0) & (steps.sum(axis=-1, keepdims=True) == 4)):
        raise ValueError("SOMA_neutral.npz triangles do not follow the USD quad winding")
    fan_diagonal = ~np.any(np.any(positions == 1, axis=-1) & np.any(positions == 3, axis=-1), axis=-1)
    LOGGER.info(
        "faces: %d triangles split the %d USD quads with matching winding; %d quads use the fan diagonal 0-2",
        len(faces),
        len(quads),
        int(fan_diagonal.sum()),
    )


def _rig_data(usd_rig: dict[str, Any], triangles: np.ndarray) -> dict[str, np.ndarray]:
    """Skin-joint rig, bind mesh and faces read from the USD template (mid skin mesh)."""
    faces = np.asarray(triangles, dtype=np.int64)
    _check_faces(faces, usd_rig["face_vert_indices"], usd_rig["face_vert_counts"])
    parents = np.asarray(usd_rig["joint_parent_ids"], dtype=np.int64)
    parents[0] = -1
    if not np.all(parents[1:] < np.arange(1, len(parents))):
        raise ValueError("USD joint order is not parent-before-child")
    dense = csc_matrix(
        (usd_rig["skinning_weights_data"], usd_rig["skinning_weights_indices"], usd_rig["skinning_weights_indptr"]),
        shape=tuple(usd_rig["skinning_weights_shape"]),
    ).toarray()
    skin_indices, skin_weights = _compact_skin(dense)
    return {
        "bind_vertices": np.asarray(usd_rig["bind_shape"], dtype=np.float32) * CENTIMETERS_TO_METERS,
        "faces": faces,
        "joint_names": np.asarray([str(name) for name in usd_rig["joint_names"]]),
        "joint_parents": parents,
        "bind_pose_world": _transforms_to_meters(usd_rig["bind_pose_world"]),
        "t_pose_world": _transforms_to_meters(usd_rig["t_pose_world"]),
        "skin_indices": skin_indices,
        "skin_weights": skin_weights,
    }


def _public_data(layer: SOMALayer) -> dict[str, np.ndarray]:
    """Public 78-joint view: ids, parents, folded skin weights and the sparse RBF joint regressor."""
    public_ids = _as_numpy(layer.public_transform_joint_indices).astype(np.int64)
    public_parents = _as_numpy(layer.public_joint_parent_ids).astype(np.int64)
    public_parents[0] = -1
    if not np.all(public_parents[1:] < np.arange(1, len(public_parents))):
        raise ValueError("Public joint order is not parent-before-child")
    public_weights = _as_numpy(layer.public_skinning_weights()).astype(np.float32)
    regressor_weights = _as_numpy(layer.skeleton_transfer.skinning_weights)
    if not np.allclose(public_weights, regressor_weights, rtol=0.0, atol=1e-6):
        raise ValueError("Folded public skin weights differ from the derived public rig weights")
    regressor = _as_numpy(layer.skeleton_transfer.sparse_rbf_matrix.to_dense()).astype(np.float32)
    if np.any(regressor[0] != 0):
        raise ValueError("Joint regressor row 0 (Root) is not zero")
    return {
        "public_joint_ids": public_ids,
        "public_parents": public_parents,
        "public_skin_weights": public_weights,
        "joint_regressor": regressor,
    }


def _twist_data(layer: SOMALayer, joint_parents: np.ndarray, public_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Twist helpers, their segments and the procedural rotation matrix, in JSON segment order."""
    procedural = layer.procedural_transforms
    if set(procedural.rotation_extraction_modes) != {TWIST_MODE}:
        raise ValueError(
            f"Expected only {TWIST_MODE} twist extraction, got {set(procedural.rotation_extraction_modes)}"
        )
    if any(segment.source_axis != 0 or segment.source_sign != 1.0 for segment in procedural.segments):
        raise ValueError("Expected every twist segment to use source axis +x")
    twist_ids = _as_numpy(procedural.twist_target_ids).astype(np.int64)
    fractions = _as_numpy(procedural.segment_fractions).astype(np.float32)
    starts = _as_numpy(procedural.segment_start_source_ids).astype(np.int64)
    ends = _as_numpy(procedural.segment_end_source_ids).astype(np.int64)
    twist_segment = np.repeat(np.arange(len(procedural.segments)), [len(s.twist_joints) for s in procedural.segments])
    start_joint_ids = public_ids[starts[twist_segment]]
    if not np.array_equal(joint_parents[twist_ids], start_joint_ids):
        raise ValueError(
            f"Twist helper parents {joint_parents[twist_ids]} differ from segment starts {start_joint_ids}"
        )
    expected_translation = np.eye(JOINT_COUNT, dtype=np.float32)
    expected_translation[twist_ids] = 0.0
    expected_translation[twist_ids, start_joint_ids] = 1.0 - fractions
    expected_translation[twist_ids, public_ids[ends[twist_segment]]] = fractions
    if not np.allclose(_as_numpy(procedural.translation_parameter_matrix), expected_translation, atol=1e-6):
        raise ValueError("Procedural translation matrix is not a start/end blend with the segment fractions")
    return {
        "twist_joint_ids": twist_ids,
        "twist_segment": twist_segment.astype(np.int64),
        "twist_fraction": fractions,
        "twist_rotation_matrix": _as_numpy(procedural.rotation_parameter_matrix).astype(np.float32),
        "segment_start": starts,
        "segment_end": ends,
        "segment_parent": _as_numpy(procedural.segment_parent_source_ids).astype(np.int64),
        "segment_reverse": _as_numpy(procedural.segment_reverse_mask).astype(bool),
    }


def _corrective_data(layer: SOMALayer) -> dict[str, np.ndarray]:
    """Effective (mask-applied) corrective MLP weights; the layer was built in centimetres."""
    module = layer.correctives_model.module
    first_layer = _as_numpy(module.W1 * module.M1_prior).astype(np.float32)
    output_layer = _as_numpy(module.W2 * module.M2_prior)
    rows, columns = np.nonzero(output_layer)
    return {
        "corrective_bind": _as_numpy(module.bindpose).astype(np.float32),
        "corrective_first_layer": first_layer,
        "corrective_output_indices": np.stack((rows, columns)).astype(np.int64),
        "corrective_output_values": output_layer[rows, columns].astype(np.float32) * CENTIMETERS_TO_METERS,
        "corrective_output_shape": np.asarray(output_layer.shape, dtype=np.int64),
        "corrective_tanh": np.asarray(bool(module.use_tanh)),
    }


def _validate(data: dict[str, np.ndarray]) -> None:
    """Check the archive contract and log the numbers the loader relies on."""
    expected_shapes = {
        "bind_vertices": (VERTEX_COUNT, 3),
        "shape_basis": (VERTEX_COUNT, 3, SHAPE_COUNT),
        "bind_pose_world": (JOINT_COUNT, 4, 4),
        "public_skin_weights": (VERTEX_COUNT, PUBLIC_JOINT_COUNT),
        "joint_regressor": (PUBLIC_JOINT_COUNT, VERTEX_COUNT),
        "twist_rotation_matrix": (TWIST_COUNT, PUBLIC_JOINT_COUNT),
        "segment_start": (SEGMENT_COUNT,),
        "corrective_first_layer": (PUBLIC_JOINT_COUNT * 6, PUBLIC_JOINT_COUNT * CORRECTIVES_PER_JOINT),
    }
    for key, shape in expected_shapes.items():
        if data[key].shape != shape:
            raise ValueError(f"{key} has shape {data[key].shape}, expected {shape}")
    if bool(data["corrective_tanh"]):
        raise ValueError("Corrective checkpoint uses tanh; the archive contract expects use_tanh=False")
    output_shape = (PUBLIC_JOINT_COUNT * CORRECTIVES_PER_JOINT, VERTEX_COUNT * 3)
    if tuple(data["corrective_output_shape"]) != output_shape:
        raise ValueError(f"corrective_output_shape {data['corrective_output_shape']} != {output_shape}")
    public_ids = data["public_joint_ids"]
    if data["joint_names"][public_ids[0]] != "Root" or data["joint_names"][public_ids[1]] != "Hips":
        raise ValueError("Public joints must start with Root, Hips")
    if not np.array_equal(data["bind_pose_world"][public_ids[0]], np.eye(4, dtype=np.float32)):
        raise ValueError("Root bind transform is not the identity")
    for key in ("skin_weights", "public_skin_weights"):
        row_error = float(np.abs(data[key].sum(axis=1) - 1.0).max())
        if row_error > 1e-5:
            raise ValueError(f"{key} rows do not sum to 1 (max error {row_error:.3g})")
    bind_joints = data["bind_pose_world"][public_ids, :3, 3]
    regressed = data["joint_regressor"].astype(np.float64) @ data["bind_vertices"]
    regressor_error = float(np.abs(regressed - bind_joints).max())
    if regressor_error > 1e-4:
        raise ValueError(f"Joint regressor misses the bind joints by {regressor_error:.3g} m")
    rotation_difference = np.abs(data["bind_pose_world"][:, :3, :3] - data["t_pose_world"][:, :3, :3]).max(axis=(1, 2))
    translation_difference = np.abs(data["bind_pose_world"][:, :3, 3] - data["t_pose_world"][:, :3, 3]).max()
    public_rotations = data["bind_pose_world"][public_ids, :3, :3]
    parent_rotations = np.swapaxes(public_rotations[np.maximum(data["public_parents"], 0)], -1, -2)
    bind_local = np.where(
        data["public_parents"][:, None, None] < 0, public_rotations, parent_rotations @ public_rotations
    )
    corrective_bind_difference = float(np.abs(data["corrective_bind"] - bind_local).max())
    if corrective_bind_difference > 1e-5:
        raise ValueError(
            f"Corrective bindpose differs from the public local bind rotations by {corrective_bind_difference}"
        )
    LOGGER.info(
        "bind vs t-pose: max rotation-entry difference %.6g (joint %s), max translation difference %.6g m",
        float(rotation_difference.max()),
        data["joint_names"][int(rotation_difference.argmax())],
        float(translation_difference),
    )
    LOGGER.info("corrective bindpose vs public local bind rotations: max difference %.3g", corrective_bind_difference)
    LOGGER.info("twist helpers: all %d parented to their segment start joint", len(data["twist_joint_ids"]))
    LOGGER.info("joint regressor vs bind joints: max error %.3g m, row 0 zero", regressor_error)
    LOGGER.info(
        "skin K=%d, corrective nonzeros: first layer %d, output %d; bind vs PCA mean max %.4g m",
        data["skin_weights"].shape[1],
        int(np.count_nonzero(data["corrective_first_layer"])),
        data["corrective_output_values"].size,
        float(np.abs(data["bind_vertices"] - data["shape_mean"]).max()),
    )


def main() -> None:
    """Convert the SOMA-X mid LOD into ``soma_mid.npz``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    for name in ASSET_FILES:
        if not (args.assets_dir / name).is_file():
            raise FileNotFoundError(f"Required SOMA asset not found: {args.assets_dir / name}")
    with torch.no_grad():
        layer = SOMALayer(
            data_root=args.assets_dir,
            device="cpu",
            identity_model_type="soma",
            mode="dense",
            output_unit=Unit.CENTIMETERS,
        )
        usd_rig = load_lod_rigs_from_usd(args.assets_dir / "SOMA_template_rig.usda", ("mid",))["mid"]
        if not np.array_equal(usd_rig["bind_pose_world"], layer.rig_data["bind_pose_world"]):
            raise ValueError("Layer rig differs from the USD mid rig")
        with np.load(args.assets_dir / "SOMA_neutral.npz", allow_pickle=False) as neutral:
            data = {
                "format_version": np.asarray(1, dtype=np.int64),
                "lod": np.asarray("mid"),
                "unit": np.asarray("m"),
                "up_axis": np.asarray("+y"),
                "forward_axis": np.asarray("+z"),
                "source": np.asarray(_source_metadata(args.assets_dir)),
                **_shape_data(neutral),
                **_rig_data(usd_rig, neutral["triangles"]),
            }
        data.update(_public_data(layer))
        data.update(_twist_data(layer, data["joint_parents"], data["public_joint_ids"]))
        data.update(_corrective_data(layer))
    _validate(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "soma_mid.npz"
    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez(temporary_path, **data)
    temporary_path.replace(output_path)
    LOGGER.info("Wrote %s (%.2f MiB)", output_path, output_path.stat().st_size / (1024 * 1024))


if __name__ == "__main__":
    main()
