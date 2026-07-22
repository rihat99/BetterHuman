"""Benchmark BetterHuman SMPL and SMPL-X against vendored upstream ``smplx``.

The upstream rows recompute shape and pose on every call. BetterHuman reports
both a shape-baked pose-only path and an apples-to-apples shape-plus-pose path.

Run the pinned CPU protocol with::

    uv run python benchmarks/benchmark_smpl.py --device cpu

Use ``--device cuda --use-warp`` to request BetterRobot's Warp FK lane for
pose-only rows. Shape-plus-pose rows intentionally retain Torch FK because
their joint placements carry the shape batch.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import gc
import importlib
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import better_robot as br
import numpy as np
from scipy.sparse import csr_matrix
import torch

import better_human as bh
from better_human.bodies import SMPLClassic, SMPLXClassic
from better_human.core import ParametricBody
from better_human.core.pipeline import lbs, skinning_transforms

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PACKAGE_ROOT = ROOT / "references" / "parametric_body" / "smplx"
DEFAULT_SMPL_PATH = ROOT / "models" / "smpl" / "SMPL_NEUTRAL.npz"
DEFAULT_SMPLX_PATH = ROOT / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
PARITY_ATOL = 1e-6
REFERENCE_VERSION = "0.1.28"


@dataclass(frozen=True)
class ModelInputs:
    """Equivalent upstream and BetterHuman inputs for one batch."""

    reference: dict[str, torch.Tensor]
    shape_groups: dict[str, torch.Tensor]
    classic: SMPLClassic | SMPLXClassic


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smpl-path", type=Path, default=DEFAULT_SMPL_PATH)
    parser.add_argument("--smplx-path", type=Path, default=DEFAULT_SMPLX_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batches", type=int, nargs="+", default=(1, 16, 256))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--use-warp",
        action="store_true",
        help="Request Warp FK for pose-only rows; shape+pose stays on Torch FK",
    )
    arguments = parser.parse_args()
    if any(batch < 1 for batch in arguments.batches):
        parser.error("--batches values must be positive")
    if arguments.warmup < 0 or arguments.iterations < 1:
        parser.error("--warmup must be nonnegative and --iterations positive")
    if arguments.use_warp and arguments.device != "cuda":
        parser.error("--use-warp requires --device cuda")
    return arguments


def _vendored_smplx() -> tuple[Any, type[Any]]:
    """Import the vendored package and return it with its ``Struct`` type."""
    sys.path.insert(0, str(REFERENCE_PACKAGE_ROOT))
    package = importlib.import_module("smplx")
    package_path = Path(package.__file__).resolve()
    if REFERENCE_PACKAGE_ROOT.resolve() not in package_path.parents:
        raise RuntimeError(f"expected vendored smplx under {REFERENCE_PACKAGE_ROOT}, got {package_path}")
    struct_type = importlib.import_module("smplx.utils").Struct
    return package, struct_type


def _smpl_data_struct(path: Path, struct_type: type[Any]) -> Any:
    """Load the converted SMPL NPZ in the format expected by upstream."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if "J_regressor" not in arrays:
        arrays["J_regressor"] = csr_matrix(
            (
                arrays.pop("J_regressor_data"),
                arrays.pop("J_regressor_indices"),
                arrays.pop("J_regressor_indptr"),
            ),
            shape=tuple(int(value) for value in arrays.pop("J_regressor_shape")),
        )
    return struct_type(**arrays)


def _reference_model(
    name: str,
    path: Path,
    package: Any,
    struct_type: type[Any],
    device: torch.device,
    batch: int,
) -> torch.nn.Module:
    """Construct an external-parameter upstream model for benchmarking."""
    common = {
        "model_path": str(path),
        "dtype": torch.float32,
        "create_betas": False,
        "create_global_orient": False,
        "create_body_pose": False,
        "create_transl": False,
        "batch_size": batch,
    }
    if name == "SMPL":
        model = package.SMPL(
            **common,
            data_struct=_smpl_data_struct(path, struct_type),
            num_betas=10,
        )
    else:
        model = package.SMPLX(
            **common,
            num_betas=16,
            num_expression_coeffs=10,
            use_pca=False,
            flat_hand_mean=False,
            use_face_contour=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_expression=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
        )
    return model.eval().to(device)


def _betterhuman_model(name: str, path: Path, device: torch.device) -> ParametricBody:
    """Construct the matching BetterHuman model without mass integration."""
    if name == "SMPL":
        return bh.SMPL(path, num_betas=10, compute_mass=False, device=device)
    return bh.SMPLX(
        path,
        num_betas=16,
        num_expression=10,
        use_hands=True,
        use_face=True,
        flat_hand_mean=False,
        compute_mass=False,
        device=device,
    )


def _smpl_inputs(batch: int, device: torch.device) -> ModelInputs:
    row_scale = torch.linspace(0.9, 1.1, batch, device=device).unsqueeze(-1)
    betas = torch.linspace(-0.25, 0.25, 10, device=device).repeat(batch, 1)
    global_orient = row_scale * torch.tensor([0.15, -0.10, 0.05], device=device)
    body_pose = row_scale * torch.linspace(-0.18, 0.18, 69, device=device).unsqueeze(0)
    transl = row_scale * torch.tensor([0.10, -0.05, 0.20], device=device)
    reference = {
        "betas": betas,
        "global_orient": global_orient,
        "body_pose": body_pose,
        "transl": transl,
    }
    classic = SMPLClassic(
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
    )
    return ModelInputs(reference=reference, shape_groups={"betas": betas}, classic=classic)


def _smplx_inputs(batch: int, device: torch.device) -> ModelInputs:
    row_scale = torch.linspace(0.9, 1.1, batch, device=device).unsqueeze(-1)
    betas = torch.linspace(-0.20, 0.20, 16, device=device).repeat(batch, 1)
    expression = torch.linspace(-0.15, 0.15, 10, device=device).repeat(batch, 1)
    global_orient = row_scale * torch.tensor([0.12, -0.08, 0.04], device=device)
    body_pose = row_scale * torch.linspace(-0.15, 0.15, 63, device=device).unsqueeze(0)
    jaw_pose = row_scale * torch.tensor([0.08, -0.04, 0.02], device=device)
    leye_pose = row_scale * torch.tensor([0.03, 0.02, -0.01], device=device)
    reye_pose = row_scale * torch.tensor([-0.02, 0.025, 0.015], device=device)
    left_hand_pose = row_scale * torch.linspace(-0.12, 0.12, 45, device=device).unsqueeze(0)
    right_hand_pose = row_scale * torch.linspace(0.10, -0.10, 45, device=device).unsqueeze(0)
    transl = row_scale * torch.tensor([0.08, -0.03, 0.12], device=device)
    reference = {
        "betas": betas,
        "expression": expression,
        "global_orient": global_orient,
        "body_pose": body_pose,
        "jaw_pose": jaw_pose,
        "leye_pose": leye_pose,
        "reye_pose": reye_pose,
        "left_hand_pose": left_hand_pose,
        "right_hand_pose": right_hand_pose,
        "transl": transl,
        "return_shaped": False,
    }
    classic = SMPLXClassic(
        global_orient=global_orient,
        body_pose=body_pose,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        expression=expression,
        transl=transl,
        num_pca_comps=None,
    )
    return ModelInputs(
        reference=reference,
        shape_groups={"betas": betas, "expression": expression},
        classic=classic,
    )


def _inputs(name: str, batch: int, device: torch.device) -> ModelInputs:
    if name == "SMPL":
        return _smpl_inputs(batch, device)
    return _smplx_inputs(batch, device)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_milliseconds(
    callback: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        callback()
    _synchronize(device)
    durations: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        callback()
        _synchronize(device)
        durations.append((time.perf_counter_ns() - started) * 1e-6)
    return statistics.median(durations)


def _vertices(body: ParametricBody, q: torch.Tensor, *, use_warp: bool) -> torch.Tensor:
    if not use_warp:
        return body.vertices(q)
    data = br.forward_kinematics(body.robot, q, use_warp=True)
    return body.vertices_from_data(data)


def _transform_and_lbs(
    body: ParametricBody,
    data: br.Data,
    corrective: torch.Tensor,
) -> torch.Tensor:
    transforms = skinning_transforms(
        body.native_joint_poses(data),
        body.values.rest_joints,
    )
    return lbs(
        body.values.v_shaped + corrective,
        transforms,
        body.values.skinning_weight_matrix,
    )


def _validate_vertices(
    name: str,
    batch: int,
    expected: torch.Tensor,
    actual: torch.Tensor,
    device: torch.device,
    path: str,
) -> float:
    expected_shape = (batch, 6890 if name == "SMPL" else 10475, 3)
    for source, vertices in (("upstream", expected), (path, actual)):
        if vertices.shape != expected_shape:
            raise RuntimeError(f"{name} {source} produced shape {tuple(vertices.shape)}; expected {expected_shape}")
        if vertices.dtype != torch.float32 or vertices.device.type != device.type or not torch.isfinite(vertices).all():
            raise RuntimeError(f"{name} {source} produced invalid dtype, device, or non-finite values")
    error = (actual - expected).abs().max().item()
    if error > PARITY_ATOL:
        raise RuntimeError(f"{name} batch {batch} {path} max error {error:.9g} exceeds {PARITY_ATOL:.1e}")
    return error


def _benchmark_model(
    name: str,
    path: Path,
    package: Any,
    struct_type: type[Any],
    batches: tuple[int, ...],
    device: torch.device,
    warmup: int,
    iterations: int,
    use_warp: bool,
) -> tuple[
    list[tuple[str, int, str, float]],
    list[tuple[str, int, str, float]],
    list[tuple[str, int, float, float]],
]:
    results: list[tuple[str, int, str, float]] = []
    breakdown: list[tuple[str, int, str, float]] = []
    agreement: list[tuple[str, int, float, float]] = []
    neutral = _betterhuman_model(name, path, device)

    for batch in batches:
        reference_model = _reference_model(name, path, package, struct_type, device, batch)
        inputs = _inputs(name, batch, device)
        scalar_shape = {group: value[0] for group, value in inputs.shape_groups.items()}
        pose_body = neutral.with_shape(**scalar_shape)
        q = pose_body.from_classic(inputs.classic)

        def reference_forward() -> torch.Tensor:
            return reference_model(**inputs.reference).vertices

        def pose_only() -> torch.Tensor:
            return _vertices(pose_body, q, use_warp=use_warp)

        def shape_and_pose() -> torch.Tensor:
            return neutral.with_shape(**inputs.shape_groups).vertices(q)

        expected = reference_forward()
        pose_vertices = pose_only()
        fair_vertices = shape_and_pose()
        pose_error = _validate_vertices(name, batch, expected, pose_vertices, device, "pose-only")
        fair_error = _validate_vertices(name, batch, expected, fair_vertices, device, "shape+pose")
        agreement.append((name, batch, pose_error, fair_error))

        results.extend(
            (
                (
                    name,
                    batch,
                    "upstream smplx shape+pose",
                    _median_milliseconds(
                        reference_forward,
                        device=device,
                        warmup=warmup,
                        iterations=iterations,
                    ),
                ),
                (
                    name,
                    batch,
                    "BetterHuman pose-only" + (" (Warp requested)" if use_warp else ""),
                    _median_milliseconds(
                        pose_only,
                        device=device,
                        warmup=warmup,
                        iterations=iterations,
                    ),
                ),
                (
                    name,
                    batch,
                    "BetterHuman shape+pose" + (" (Torch FK)" if use_warp else ""),
                    _median_milliseconds(
                        shape_and_pose,
                        device=device,
                        warmup=warmup,
                        iterations=iterations,
                    ),
                ),
            )
        )

        data = br.forward_kinematics(pose_body.robot, q, use_warp=use_warp)
        corrective = pose_body._pose_corrective_policy(pose_body, data)
        breakdown.extend(
            (
                (
                    name,
                    batch,
                    "FK" + (" (Warp requested)" if use_warp else ""),
                    _median_milliseconds(
                        lambda: br.forward_kinematics(pose_body.robot, q, use_warp=use_warp),
                        device=device,
                        warmup=warmup,
                        iterations=iterations,
                    ),
                ),
                (
                    name,
                    batch,
                    "corrective",
                    _median_milliseconds(
                        lambda: pose_body._pose_corrective_policy(pose_body, data),
                        device=device,
                        warmup=warmup,
                        iterations=iterations,
                    ),
                ),
                (
                    name,
                    batch,
                    "transforms + LBS",
                    _median_milliseconds(
                        lambda: _transform_and_lbs(pose_body, data, corrective),
                        device=device,
                        warmup=warmup,
                        iterations=iterations,
                    ),
                ),
            )
        )

        reference_forward = pose_only = shape_and_pose = None
        expected = pose_vertices = fair_vertices = reference_model = None
        inputs = scalar_shape = pose_body = q = data = corrective = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    neutral = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results, breakdown, agreement


def _print_end_to_end(rows: list[tuple[str, int, str, float]]) -> None:
    references = {(name, batch): median for name, batch, path, median in rows if path == "upstream smplx shape+pose"}
    print("\n## End-to-end\n")
    print("| Model | Batch | Path | Median ms | x original |")
    print("|---|---:|---|---:|---:|")
    for name, batch, path, median in rows:
        ratio = median / references[(name, batch)]
        print(f"| {name} | {batch} | {path} | {median:.6f} | {ratio:.6f} |")


def _print_breakdown(rows: list[tuple[str, int, str, float]]) -> None:
    print("\n## BetterHuman pose-only breakdown\n")
    print("| Model | Batch | Stage | Median ms |")
    print("|---|---:|---|---:|")
    for name, batch, stage, median in rows:
        print(f"| {name} | {batch} | {stage} | {median:.6f} |")


def _print_agreement(rows: list[tuple[str, int, float, float]]) -> None:
    print("\n## Pre-timing output agreement\n")
    print("| Model | Batch | Pose-only max abs m | Shape+pose max abs m |")
    print("|---|---:|---:|---:|")
    for name, batch, pose_error, fair_error in rows:
        print(f"| {name} | {batch} | {pose_error:.17g} | {fair_error:.17g} |")


def main() -> None:
    """Run the requested benchmark grid and print copyable Markdown tables."""
    arguments = _arguments()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    for name, path in (("SMPL", arguments.smpl_path), ("SMPL-X", arguments.smplx_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} model archive is absent: {path}")

    package, struct_type = _vendored_smplx()
    batches = tuple(arguments.batches)
    print(f"torch={torch.__version__}")
    print(f"better_robot={br.__version__}")
    print(f"reference_version={REFERENCE_VERSION}")
    print(f"reference_package={Path(package.__file__).resolve().parent}")
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"cpu_threads={torch.get_num_threads()}")
    print(f"cpu_interop_threads={torch.get_num_interop_threads()}")
    print(f"warmup={arguments.warmup}; iterations={arguments.iterations}; batches={batches}")
    print(f"use_warp={arguments.use_warp}")
    print(f"smpl_path={arguments.smpl_path}")
    print(f"smplx_path={arguments.smplx_path}")

    results: list[tuple[str, int, str, float]] = []
    breakdown: list[tuple[str, int, str, float]] = []
    agreement: list[tuple[str, int, float, float]] = []
    with torch.inference_mode():
        for name, path in (("SMPL", arguments.smpl_path), ("SMPL-X", arguments.smplx_path)):
            model_results, model_breakdown, model_agreement = _benchmark_model(
                name,
                path,
                package,
                struct_type,
                batches,
                device,
                arguments.warmup,
                arguments.iterations,
                arguments.use_warp,
            )
            results.extend(model_results)
            breakdown.extend(model_breakdown)
            agreement.extend(model_agreement)

    _print_agreement(agreement)
    _print_end_to_end(results)
    _print_breakdown(breakdown)


if __name__ == "__main__":
    main()
