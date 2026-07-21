"""Dump deterministic float32 MHR parity goldens from the reference package."""

from __future__ import annotations

import argparse
import gc
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

LOD_COUNT = 7
LOD1_CASE_COUNT = 30
SMOKE_CASE_COUNT = 3
IDENTITY_COUNT = 45
MODEL_PARAMETER_COUNT = 204
EXPRESSION_COUNT = 72
JOINT_COUNT = 127
SKELETON_STATE_WIDTH = 8
VERTEX_COUNTS = (73639, 18439, 10661, 4899, 2461, 971, 595)
BASE_SEED = 20260721
ROOT_PARAMETER_NAMES = ("root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz")
HAND_SCALE_NAMES = ("scale_l_hands", "scale_r_hands")

LOGGER = logging.getLogger(__name__)


class _ParameterMetadata(NamedTuple):
    names: np.ndarray
    channel_types: np.ndarray
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    tpose_values: np.ndarray
    tpose_mask: np.ndarray
    native_unit: str
    native_pose_format: str
    length_scale_to_meters: np.float32


class _ParameterMasks(NamedTuple):
    pose: np.ndarray
    proportions: np.ndarray
    hand_scale: np.ndarray
    non_root: np.ndarray


class _Case(NamedTuple):
    identity_coeffs: np.ndarray
    model_parameters: np.ndarray
    face_expression: np.ndarray | None
    apply_correctives: bool
    seed: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("models/MHR/assets"),
        help="Directory containing the original MHR release assets",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("references/parametric_body/MHR"),
        help="Read-only checkout containing the reference mhr package",
    )
    parser.add_argument(
        "--converted-dir",
        type=Path,
        default=Path("models/MHR/converted"),
        help="Directory containing converter archives; LOD1 supplies parameter metadata",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/parity/goldens"),
        help="Parent directory for mhr_lodN golden directories",
    )
    parser.add_argument(
        "--lod",
        action="append",
        choices=range(LOD_COUNT),
        type=int,
        help="Dump one LOD; repeat for several. Defaults to all seven.",
    )
    return parser.parse_args()


def _load_parameter_metadata(converted_dir: Path) -> _ParameterMetadata:
    archive_path = converted_dir / "mhr_lod1.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(f"Converted LOD1 metadata not found at {archive_path}; run the MHR converter first")
    with np.load(archive_path, allow_pickle=False) as archive:
        metadata = _ParameterMetadata(
            names=np.asarray(archive["model_parameter_names"]),
            channel_types=np.asarray(archive["model_parameter_channel_types"], dtype=np.int8),
            lower_limits=np.asarray(archive["model_parameter_lower_limits"], dtype=np.float32),
            upper_limits=np.asarray(archive["model_parameter_upper_limits"], dtype=np.float32),
            tpose_values=np.asarray(archive["tpose_values"], dtype=np.float32),
            tpose_mask=np.asarray(archive["tpose_mask"], dtype=np.bool_),
            native_unit=str(np.asarray(archive["native_unit"]).item()),
            native_pose_format=str(np.asarray(archive["native_pose_format"]).item()),
            length_scale_to_meters=np.float32(np.asarray(archive["length_scale_to_meters"]).item()),
        )
    expected_shape = (MODEL_PARAMETER_COUNT,)
    for name, values in (
        ("names", metadata.names),
        ("channel types", metadata.channel_types),
        ("lower limits", metadata.lower_limits),
        ("upper limits", metadata.upper_limits),
        ("tpose values", metadata.tpose_values),
        ("tpose mask", metadata.tpose_mask),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"Expected {name} shape {expected_shape}, got {values.shape}")
    if tuple(metadata.names[:6].tolist()) != ROOT_PARAMETER_NAMES:
        raise ValueError(f"Unexpected MHR root parameter order: {metadata.names[:6].tolist()}")
    if metadata.native_unit != "centimeter" or metadata.native_pose_format != "euler_xyz":
        raise ValueError(f"Unexpected MHR conventions: unit={metadata.native_unit}, pose={metadata.native_pose_format}")
    if metadata.length_scale_to_meters != np.float32(0.01):
        raise ValueError(f"Unexpected MHR length scale {metadata.length_scale_to_meters}")
    return metadata


def _parameter_masks(metadata: _ParameterMetadata) -> _ParameterMasks:
    root = np.zeros(MODEL_PARAMETER_COUNT, dtype=np.bool_)
    root[:6] = True
    proportions = (metadata.channel_types == 0) & ~root
    pose = (metadata.channel_types == 1) & ~root
    hand_scale = metadata.channel_types == 2
    non_root = ~root
    counts = tuple(int(mask.sum()) for mask in (pose, proportions, hand_scale, non_root))
    if counts != (125, 71, 2, 198):
        raise ValueError(f"Expected pose/proportion/hand-scale/non-root counts (125, 71, 2, 198), got {counts}")
    if set(metadata.names[hand_scale].tolist()) != set(HAND_SCALE_NAMES):
        raise ValueError(f"Unexpected true-scale parameters: {metadata.names[hand_scale].tolist()}")
    if not np.all(np.isfinite(metadata.lower_limits[non_root])) or not np.all(
        np.isfinite(metadata.upper_limits[non_root])
    ):
        raise ValueError("Every non-root MHR parameter must have finite sampling limits")
    if np.any(metadata.lower_limits > metadata.upper_limits):
        raise ValueError("MHR parameter limits contain a lower bound above its upper bound")
    if int(metadata.tpose_mask.sum()) != 19:
        raise ValueError(f"Expected 19 authored tpose values, got {int(metadata.tpose_mask.sum())}")
    name_to_index = {name: index for index, name in enumerate(metadata.names.tolist())}
    for elbow_name in ("l_elbow_bend", "r_elbow_bend"):
        if metadata.tpose_values[name_to_index[elbow_name]] != np.float32(-0.5):
            raise ValueError(f"Unexpected tpose value for {elbow_name}")
    return _ParameterMasks(pose=pose, proportions=proportions, hand_scale=hand_scale, non_root=non_root)


def _random_coefficients(generator: np.random.Generator, count: int) -> np.ndarray:
    return generator.uniform(-2.0, 2.0, size=count).astype(np.float32)


def _sample_parameters(
    generator: np.random.Generator,
    metadata: _ParameterMetadata,
    mask: np.ndarray,
) -> np.ndarray:
    values = np.zeros(MODEL_PARAMETER_COUNT, dtype=np.float32)
    lower = metadata.lower_limits[mask]
    upper = metadata.upper_limits[mask]
    samples = lower + generator.random(int(mask.sum())).astype(np.float32) * (upper - lower)
    values[mask] = samples
    return values


def _case(
    seed: int,
    *,
    identity_coeffs: np.ndarray | None = None,
    model_parameters: np.ndarray | None = None,
    face_expression: np.ndarray | None = None,
    expression_present: bool = True,
    apply_correctives: bool = True,
) -> _Case:
    identity = np.zeros(IDENTITY_COUNT, dtype=np.float32) if identity_coeffs is None else np.asarray(identity_coeffs)
    parameters = (
        np.zeros(MODEL_PARAMETER_COUNT, dtype=np.float32) if model_parameters is None else np.asarray(model_parameters)
    )
    expression = None
    if expression_present:
        expression = (
            np.zeros(EXPRESSION_COUNT, dtype=np.float32) if face_expression is None else np.asarray(face_expression)
        )
    for name, values, expected_shape in (
        ("identity coefficients", identity, (IDENTITY_COUNT,)),
        ("model parameters", parameters, (MODEL_PARAMETER_COUNT,)),
        ("face expression", expression, (EXPRESSION_COUNT,)),
    ):
        if values is not None and (values.shape != expected_shape or values.dtype != np.float32):
            raise ValueError(f"Expected float32 {name} with shape {expected_shape}, got {values.dtype} {values.shape}")
    return _Case(identity, parameters, expression, apply_correctives, seed)


def _root_parameters(generator: np.random.Generator) -> np.ndarray:
    parameters = np.zeros(MODEL_PARAMETER_COUNT, dtype=np.float32)
    parameters[:3] = generator.uniform(-0.25, 0.25, size=3).astype(np.float32)
    parameters[3:6] = generator.uniform(-1.0, 1.0, size=3).astype(np.float32)
    return parameters


def _combined_case(
    seed: int,
    metadata: _ParameterMetadata,
    masks: _ParameterMasks,
    *,
    expression_present: bool = True,
    apply_correctives: bool = True,
) -> _Case:
    generator = np.random.default_rng(seed)
    model_parameters = _sample_parameters(generator, metadata, masks.non_root)
    model_parameters[:6] = _root_parameters(generator)[:6]
    return _case(
        seed,
        identity_coeffs=_random_coefficients(generator, IDENTITY_COUNT),
        model_parameters=model_parameters,
        face_expression=_random_coefficients(generator, EXPRESSION_COUNT),
        expression_present=expression_present,
        apply_correctives=apply_correctives,
    )


def _lod1_cases(metadata: _ParameterMetadata, masks: _ParameterMasks) -> dict[str, _Case]:
    seed_base = BASE_SEED + 1000
    cases: dict[str, _Case] = {
        "zero_correctives_on": _case(seed_base),
        "zero_correctives_off": _case(seed_base + 1, apply_correctives=False),
        "tpose": _case(seed_base + 2, model_parameters=metadata.tpose_values.copy()),
    }
    for index in range(5):
        seed = seed_base + 3 + index
        parameters = _sample_parameters(np.random.default_rng(seed), metadata, masks.pose)
        cases[f"random_pose_{index:02d}"] = _case(seed, model_parameters=parameters)
    for index in range(3):
        seed = seed_base + 8 + index
        identity = _random_coefficients(np.random.default_rng(seed), IDENTITY_COUNT)
        cases[f"identity_{index:02d}"] = _case(seed, identity_coeffs=identity)
    for index in range(3):
        seed = seed_base + 11 + index
        expression = _random_coefficients(np.random.default_rng(seed), EXPRESSION_COUNT)
        cases[f"expression_{index:02d}"] = _case(seed, face_expression=expression)
    cases["expression_none"] = _combined_case(
        seed_base + 14,
        metadata,
        masks,
        expression_present=False,
    )
    for index in range(3):
        seed = seed_base + 15 + index
        parameters = _sample_parameters(np.random.default_rng(seed), metadata, masks.proportions)
        cases[f"proportions_{index:02d}"] = _case(seed, model_parameters=parameters)

    name_to_index = {name: index for index, name in enumerate(metadata.names.tolist())}
    left_index, right_index = (name_to_index[name] for name in HAND_SCALE_NAMES)
    for name, left_value, right_value, seed_offset in (
        ("hand_scale_min", metadata.lower_limits[left_index], metadata.lower_limits[right_index], 18),
        ("hand_scale_max", metadata.upper_limits[left_index], metadata.upper_limits[right_index], 19),
        ("hand_scale_crossed", metadata.lower_limits[left_index], metadata.upper_limits[right_index], 20),
    ):
        parameters = np.zeros(MODEL_PARAMETER_COUNT, dtype=np.float32)
        parameters[left_index] = left_value
        parameters[right_index] = right_value
        cases[name] = _case(seed_base + seed_offset, model_parameters=parameters)

    root_values = (
        np.asarray((0.125, -0.075, 0.05, 0.8, -0.6, 0.4), dtype=np.float32),
        np.asarray((-0.2, 0.15, -0.1, -0.45, 0.35, -0.25), dtype=np.float32),
    )
    for index, values in enumerate(root_values):
        parameters = np.zeros(MODEL_PARAMETER_COUNT, dtype=np.float32)
        parameters[:6] = values
        cases[f"root_transform_{index:02d}"] = _case(seed_base + 21 + index, model_parameters=parameters)
    for index in range(7):
        seed = seed_base + 23 + index
        cases[f"combined_{index:02d}"] = _combined_case(
            seed,
            metadata,
            masks,
            apply_correctives=index != 6,
        )
    if len(cases) != LOD1_CASE_COUNT:
        raise RuntimeError(f"Expected {LOD1_CASE_COUNT} LOD1 cases, built {len(cases)}")
    return cases


def _smoke_cases(lod: int, metadata: _ParameterMetadata, masks: _ParameterMasks) -> dict[str, _Case]:
    seed_base = BASE_SEED + lod * 1000
    random_seed = seed_base + 1
    proportion_seed = seed_base + 2
    generator = np.random.default_rng(proportion_seed)
    proportion_parameters = _sample_parameters(generator, metadata, masks.proportions | masks.hand_scale)
    cases = {
        "zero": _case(seed_base),
        "fully_random": _combined_case(random_seed, metadata, masks),
        "proportions_hand_scale": _case(proportion_seed, model_parameters=proportion_parameters),
    }
    if len(cases) != SMOKE_CASE_COUNT:
        raise RuntimeError(f"Expected {SMOKE_CASE_COUNT} LOD{lod} cases, built {len(cases)}")
    return cases


def _load_reference_model(reference_dir: Path, assets_dir: Path, lod: int) -> Any:
    package_file = reference_dir / "mhr" / "mhr.py"
    if not package_file.is_file():
        raise FileNotFoundError(f"Reference MHR package not found at {package_file}")
    reference_path = str(reference_dir.resolve())
    if reference_path not in sys.path:
        sys.path.insert(0, reference_path)
    sys.dont_write_bytecode = True
    module = importlib.import_module("mhr.mhr")
    loaded_path = Path(module.__file__).resolve()
    if not loaded_path.is_relative_to(reference_dir.resolve()):
        raise ImportError(f"Imported MHR reference from {loaded_path}, expected it under {reference_dir.resolve()}")
    model = module.MHR.from_files(
        folder=assets_dir.resolve(),
        device=torch.device("cpu"),
        lod=lod,
        wants_pose_correctives=True,
    )
    model.eval()
    return model


def _evaluate(model: Any, case: _Case, lod: int) -> tuple[np.ndarray, np.ndarray]:
    identity = torch.from_numpy(case.identity_coeffs).unsqueeze(0)
    parameters = torch.from_numpy(case.model_parameters).unsqueeze(0)
    expression = None if case.face_expression is None else torch.from_numpy(case.face_expression).unsqueeze(0)
    vertices, skel_state = model(
        identity_coeffs=identity,
        model_parameters=parameters,
        face_expr_coeffs=expression,
        apply_correctives=case.apply_correctives,
    )
    if vertices.dtype != torch.float32 or skel_state.dtype != torch.float32:
        raise TypeError(f"The MHR oracle must remain float32, got {vertices.dtype} and {skel_state.dtype}")
    expected_vertices_shape = (1, VERTEX_COUNTS[lod], 3)
    expected_skeleton_shape = (1, JOINT_COUNT, SKELETON_STATE_WIDTH)
    if tuple(vertices.shape) != expected_vertices_shape or tuple(skel_state.shape) != expected_skeleton_shape:
        raise ValueError(
            f"Unexpected LOD{lod} oracle shapes: vertices={tuple(vertices.shape)}, skel_state={tuple(skel_state.shape)}"
        )
    return vertices[0].detach().cpu().numpy(), skel_state[0].detach().cpu().numpy()


def _write_case(
    output_path: Path,
    name: str,
    lod: int,
    case: _Case,
    metadata: _ParameterMetadata,
    vertices: np.ndarray,
    skel_state: np.ndarray,
) -> None:
    face_expression = (
        np.zeros(EXPRESSION_COUNT, dtype=np.float32) if case.face_expression is None else case.face_expression
    )
    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        format_version=np.asarray(1, dtype=np.int64),
        case_name=np.asarray(name, dtype=np.str_),
        seed=np.asarray(case.seed, dtype=np.int64),
        lod=np.asarray(lod, dtype=np.int64),
        oracle=np.asarray("MHR reference package via pymomentum", dtype=np.str_),
        oracle_dtype=np.asarray("float32", dtype=np.str_),
        native_unit=np.asarray(metadata.native_unit, dtype=np.str_),
        native_pose_format=np.asarray(metadata.native_pose_format, dtype=np.str_),
        length_scale_to_meters=np.asarray(metadata.length_scale_to_meters, dtype=np.float32),
        skeleton_state_format=np.asarray("translation_xyz_quaternion_xyzw_scale", dtype=np.str_),
        identity_coeffs=case.identity_coeffs,
        model_parameters=case.model_parameters,
        face_expression=face_expression,
        face_expression_present=np.asarray(case.face_expression is not None, dtype=np.bool_),
        apply_correctives=np.asarray(case.apply_correctives, dtype=np.bool_),
        vertices=vertices,
        skel_state=skel_state,
    )
    temporary_path.replace(output_path)


def _self_check(model: Any, golden_path: Path, lod: int) -> None:
    with np.load(golden_path, allow_pickle=False) as golden:
        face_expression = np.asarray(golden["face_expression"], dtype=np.float32)
        case = _Case(
            identity_coeffs=np.asarray(golden["identity_coeffs"], dtype=np.float32),
            model_parameters=np.asarray(golden["model_parameters"], dtype=np.float32),
            face_expression=face_expression if bool(golden["face_expression_present"]) else None,
            apply_correctives=bool(golden["apply_correctives"]),
            seed=int(golden["seed"]),
        )
        expected_vertices = np.asarray(golden["vertices"], dtype=np.float32)
        expected_skel_state = np.asarray(golden["skel_state"], dtype=np.float32)
    vertices, skel_state = _evaluate(model, case, lod)
    if not np.array_equal(vertices, expected_vertices) or not np.array_equal(skel_state, expected_skel_state):
        raise RuntimeError(f"MHR golden self-check was not bitwise reproducible: {golden_path}")
    LOGGER.info("Bitwise self-check passed for %s", golden_path)


def main() -> None:
    """Evaluate the fixed MHR grids and write one compressed archive per case."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    lods = sorted(set(args.lod if args.lod is not None else range(LOD_COUNT)))
    metadata = _load_parameter_metadata(args.converted_dir)
    masks = _parameter_masks(metadata)
    torch.manual_seed(BASE_SEED)
    torch.use_deterministic_algorithms(True)
    self_check_lod = max(lods)

    for lod in lods:
        cases = _lod1_cases(metadata, masks) if lod == 1 else _smoke_cases(lod, metadata, masks)
        output_dir = args.output_dir / f"mhr_lod{lod}"
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Loading the float32 CPU MHR reference for LOD%d", lod)
        model = _load_reference_model(args.reference_dir, args.assets_dir, lod)
        with torch.no_grad():
            for name, case in cases.items():
                vertices, skel_state = _evaluate(model, case, lod)
                _write_case(output_dir / f"{name}.npz", name, lod, case, metadata, vertices, skel_state)
            if lod == self_check_lod:
                check_name = "combined_00" if lod == 1 else "fully_random"
                _self_check(model, output_dir / f"{check_name}.npz", lod)
        LOGGER.info("Wrote %d LOD%d goldens to %s", len(cases), lod, output_dir)
        del model
        gc.collect()


if __name__ == "__main__":
    main()
