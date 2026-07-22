"""Benchmark BetterHuman SMPL and SMPL-X against vendored upstream ``smplx``.

Workload A compares full shape-plus-pose forwards. Workload B fixes shape once
and streams poses; upstream still performs its full forward on every frame
because it has no shape-amortized API.

Run the pinned CPU protocol with::

    uv run python benchmarks/benchmark_smpl.py --device cpu

Use ``--device cuda --use-warp`` to request BetterRobot's Warp FK lane for
Workload B. Workload A intentionally retains Torch FK because its joint
placements carry the shape batch. Add ``--grad`` to measure forward and
forward-plus-backward medians for both workloads.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import gc
import importlib
from pathlib import Path
import statistics
import subprocess
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
        "--grad",
        action="store_true",
        help="Also benchmark forward and forward-plus-backward passes",
    )
    parser.add_argument(
        "--use-warp",
        action="store_true",
        help="Request Warp FK for Workload B; Workload A stays on Torch FK",
    )
    arguments = parser.parse_args()
    if any(batch < 1 for batch in arguments.batches):
        parser.error("--batches values must be positive")
    if arguments.warmup < 0 or arguments.iterations < 1:
        parser.error("--warmup must be nonnegative and --iterations positive")
    if arguments.use_warp and arguments.device != "cuda":
        parser.error("--use-warp requires --device cuda")
    return arguments


def _cuda_driver_version() -> str:
    """Return the NVIDIA driver reported by the benchmark host."""
    result = subprocess.run(
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()[0].strip()


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
    list[tuple[str, int, str, float]],
    list[tuple[str, int, float, float]],
]:
    workload_a: list[tuple[str, int, str, float]] = []
    workload_b: list[tuple[str, int, str, float]] = []
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

        def streaming_forward() -> torch.Tensor:
            return _vertices(pose_body, q, use_warp=use_warp)

        def full_forward() -> torch.Tensor:
            return neutral(q, **inputs.shape_groups)

        expected = reference_forward()
        streaming_vertices = streaming_forward()
        full_vertices = full_forward()
        streaming_error = _validate_vertices(
            name,
            batch,
            expected,
            streaming_vertices,
            device,
            "Workload B streaming poses",
        )
        full_error = _validate_vertices(
            name,
            batch,
            expected,
            full_vertices,
            device,
            "Workload A full forward",
        )
        agreement.append((name, batch, full_error, streaming_error))

        reference_median = _median_milliseconds(
            reference_forward,
            device=device,
            warmup=warmup,
            iterations=iterations,
        )
        workload_a.extend(
            (
                (
                    name,
                    batch,
                    "upstream smplx full forward",
                    reference_median,
                ),
                (
                    name,
                    batch,
                    "BetterHuman one-call full forward" + (" (Torch FK)" if use_warp else ""),
                    _median_milliseconds(
                        full_forward,
                        device=device,
                        warmup=warmup,
                        iterations=iterations,
                    ),
                ),
            )
        )
        workload_b.extend(
            (
                (
                    name,
                    batch,
                    "upstream smplx full forward",
                    reference_median,
                ),
                (
                    name,
                    batch,
                    "BetterHuman shape-baked vertices" + (" (Warp requested)" if use_warp else ""),
                    _median_milliseconds(
                        streaming_forward,
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

        reference_forward = streaming_forward = full_forward = None
        expected = streaming_vertices = full_vertices = reference_model = None
        inputs = scalar_shape = pose_body = q = data = corrective = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    neutral = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return workload_a, workload_b, breakdown, agreement


def _gradient_reference_inputs(
    inputs: ModelInputs,
    *,
    shape_requires_grad: bool,
) -> dict[str, Any]:
    shape_names = set(inputs.shape_groups)
    return {
        name: (
            value.detach()
            .clone()
            .requires_grad_(shape_requires_grad or name not in shape_names)
            if isinstance(value, torch.Tensor)
            else value
        )
        for name, value in inputs.reference.items()
    }


def _clear_gradients(tensors: list[torch.Tensor]) -> None:
    for tensor in tensors:
        tensor.grad = None


def _gradient_medians(
    forward: Callable[[], torch.Tensor],
    leaves: list[torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    def forward_backward() -> None:
        forward().sum().backward()

    for _ in range(warmup):
        forward()
        _synchronize(device)
        _clear_gradients(leaves)
        forward_backward()
        _synchronize(device)

    forward_durations: list[float] = []
    forward_backward_durations: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        forward()
        _synchronize(device)
        forward_durations.append((time.perf_counter_ns() - started) * 1e-6)

        _clear_gradients(leaves)
        started = time.perf_counter_ns()
        forward_backward()
        _synchronize(device)
        forward_backward_durations.append((time.perf_counter_ns() - started) * 1e-6)

    return statistics.median(forward_durations), statistics.median(forward_backward_durations)


def _benchmark_gradients(
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
    list[tuple[str, int, str, float, float]],
    list[tuple[str, int, str, float, float]],
]:
    workload_a: list[tuple[str, int, str, float, float]] = []
    workload_b: list[tuple[str, int, str, float, float]] = []
    neutral = _betterhuman_model(name, path, device)

    for batch in batches:
        reference_model = _reference_model(name, path, package, struct_type, device, batch)
        reference_model.requires_grad_(False)
        inputs = _inputs(name, batch, device)
        scalar_shape = {group: value[0] for group, value in inputs.shape_groups.items()}
        pose_body = neutral.with_shape(**scalar_shape)
        q_value = pose_body.from_classic(inputs.classic).detach()

        q_full = q_value.clone().requires_grad_()
        full_shape = {
            group: value.detach().clone().requires_grad_()
            for group, value in inputs.shape_groups.items()
        }
        full_leaves = [q_full, *full_shape.values()]
        upstream_full_inputs = _gradient_reference_inputs(inputs, shape_requires_grad=True)
        upstream_full_leaves = [
            value for value in upstream_full_inputs.values() if isinstance(value, torch.Tensor)
        ]

        def betterhuman_full_forward() -> torch.Tensor:
            return neutral(q_full, **full_shape)

        def upstream_full_forward() -> torch.Tensor:
            return reference_model(**upstream_full_inputs).vertices

        q_streaming = q_value.clone().requires_grad_()
        upstream_streaming_inputs = _gradient_reference_inputs(inputs, shape_requires_grad=False)
        upstream_streaming_leaves = [
            value
            for group, value in upstream_streaming_inputs.items()
            if group not in inputs.shape_groups and isinstance(value, torch.Tensor)
        ]

        def betterhuman_streaming_forward() -> torch.Tensor:
            return _vertices(pose_body, q_streaming, use_warp=use_warp)

        def upstream_streaming_forward() -> torch.Tensor:
            return reference_model(**upstream_streaming_inputs).vertices

        warp_gradient_suffix = (
            " (Warp forward; Torch-recompute VJP)" if use_warp else ""
        )

        gradient_cases = (
            (workload_a, "upstream smplx full forward", upstream_full_forward, upstream_full_leaves),
            (
                workload_a,
                "BetterHuman one-call full forward" + (" (Torch FK)" if use_warp else ""),
                betterhuman_full_forward,
                full_leaves,
            ),
            (
                workload_b,
                "upstream smplx full forward",
                upstream_streaming_forward,
                upstream_streaming_leaves,
            ),
            (
                workload_b,
                "BetterHuman shape-baked vertices" + warp_gradient_suffix,
                betterhuman_streaming_forward,
                [q_streaming],
            ),
        )
        for rows, label, forward, leaves in gradient_cases:
            rows.append(
                (
                    name,
                    batch,
                    label,
                    *_gradient_medians(
                        forward,
                        leaves,
                        device=device,
                        warmup=warmup,
                        iterations=iterations,
                    ),
                )
            )

        betterhuman_full_forward = upstream_full_forward = None
        betterhuman_streaming_forward = upstream_streaming_forward = None
        gradient_cases = None
        reference_model = inputs = scalar_shape = pose_body = q_value = None
        q_full = full_shape = full_leaves = upstream_full_inputs = upstream_full_leaves = None
        q_streaming = upstream_streaming_inputs = upstream_streaming_leaves = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    neutral = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return workload_a, workload_b


def _print_workload(
    title: str,
    rows: list[tuple[str, int, str, float]],
    *,
    note: str,
) -> None:
    references = {
        (name, batch): median
        for name, batch, path, median in rows
        if path == "upstream smplx full forward"
    }
    print(f"\n## {title}\n")
    print(note)
    print("\n| Model | Batch | Implementation | Median ms | × upstream |")
    print("|---|---:|---|---:|---:|")
    for name, batch, path, median in rows:
        ratio = median / references[(name, batch)]
        print(f"| {name} | {batch} | {path} | {median:.6f} | {ratio:.6f} |")


def _print_gradients(
    title: str,
    rows: list[tuple[str, int, str, float, float]],
    *,
    note: str,
) -> None:
    references = {
        (name, batch): forward_backward
        for name, batch, path, _, forward_backward in rows
        if path == "upstream smplx full forward"
    }
    print(f"\n## Gradients: {title}\n")
    print(note)
    print("F+B / F is the training-step multiplier, not a backward-only median.")
    print(
        "\n| Model | Batch | Implementation | Forward ms | Forward + backward ms | "
        "F+B / F | F+B × upstream |"
    )
    print("|---|---:|---|---:|---:|---:|---:|")
    for name, batch, path, forward, forward_backward in rows:
        print(
            f"| {name} | {batch} | {path} | {forward:.6f} | "
            f"{forward_backward:.6f} | {forward_backward / forward:.6f} | "
            f"{forward_backward / references[(name, batch)]:.6f} |"
        )


def _print_breakdown(rows: list[tuple[str, int, str, float]]) -> None:
    print("\n## BetterHuman Workload B stage breakdown\n")
    print("| Model | Batch | Stage | Median ms |")
    print("|---|---:|---|---:|")
    for name, batch, stage, median in rows:
        print(f"| {name} | {batch} | {stage} | {median:.6f} |")


def _print_agreement(rows: list[tuple[str, int, float, float]]) -> None:
    print("\n## Pre-timing output agreement\n")
    print("| Model | Batch | Workload A max abs m | Workload B max abs m |")
    print("|---|---:|---:|---:|")
    for name, batch, full_error, streaming_error in rows:
        print(f"| {name} | {batch} | {full_error:.17g} | {streaming_error:.17g} |")


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
    if device.type == "cuda":
        print(f"cuda_runtime={torch.version.cuda}")
        print(f"cuda_driver={_cuda_driver_version()}")
    print(f"cpu_threads={torch.get_num_threads()}")
    print(f"cpu_interop_threads={torch.get_num_interop_threads()}")
    print(f"warmup={arguments.warmup}; iterations={arguments.iterations}; batches={batches}")
    print(f"use_warp={arguments.use_warp}")
    print(f"grad={arguments.grad}")
    print("timing_caveat=shared-host scheduling and concurrent load can affect medians")
    print(f"smpl_path={arguments.smpl_path}")
    print(f"smplx_path={arguments.smplx_path}")

    workload_a: list[tuple[str, int, str, float]] = []
    workload_b: list[tuple[str, int, str, float]] = []
    breakdown: list[tuple[str, int, str, float]] = []
    agreement: list[tuple[str, int, float, float]] = []
    with torch.inference_mode():
        for name, path in (("SMPL", arguments.smpl_path), ("SMPL-X", arguments.smplx_path)):
            model_workload_a, model_workload_b, model_breakdown, model_agreement = _benchmark_model(
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
            workload_a.extend(model_workload_a)
            workload_b.extend(model_workload_b)
            breakdown.extend(model_breakdown)
            agreement.extend(model_agreement)

    _print_agreement(agreement)
    _print_workload(
        "Workload A — full forward (new shape + new pose every call)",
        workload_a,
        note=(
            "The × upstream ratio compares equivalent full shape-plus-pose computations; "
            "BetterHuman uses its one-call API."
        ),
    )
    _print_workload(
        "Workload B — streaming poses (shape fixed, poses vary per call)",
        workload_b,
        note=(
            "BetterHuman bakes shape once outside timing. Upstream has no shape-amortized API, "
            "so its per-frame workload remains a full forward. This is a workload-level ratio, "
            "not a FLOP-matched ratio."
        ),
    )
    _print_breakdown(breakdown)

    if arguments.grad:
        gradient_a: list[tuple[str, int, str, float, float]] = []
        gradient_b: list[tuple[str, int, str, float, float]] = []
        for name, path in (("SMPL", arguments.smpl_path), ("SMPL-X", arguments.smplx_path)):
            model_gradient_a, model_gradient_b = _benchmark_gradients(
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
            gradient_a.extend(model_gradient_a)
            gradient_b.extend(model_gradient_b)
        _print_gradients(
            "Workload A — full forward",
            gradient_a,
            note=(
                "Loss is vertices.sum(). BetterHuman differentiates q and every shape group; "
                "upstream differentiates every supplied pose and shape tensor."
            ),
        )
        _print_gradients(
            "Workload B — streaming poses",
            gradient_b,
            note=(
                "Loss is vertices.sum(). BetterHuman differentiates q after baking shape outside "
                "timing. Upstream differentiates pose inputs but must still run its full forward."
            ),
        )


if __name__ == "__main__":
    main()
