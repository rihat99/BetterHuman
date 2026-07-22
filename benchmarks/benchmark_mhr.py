"""Benchmark two MHR workloads against the released LOD1 TorchScript model.

Workload A changes shape and pose every call, so both implementations run a
full forward pass. Workload B fixes shape and streams poses; BetterHuman bakes
shape once, while the reference must still run its full forward API per frame.

Run the pinned protocol on CPU with::

    uv run python benchmarks/benchmark_mhr.py --device cpu

Use ``--device cuda --use-warp`` to exercise BetterRobot's fused FK lane for
the streaming-pose rows. Batched shape values keep full-forward rows on the
Torch FK lane. Use ``--grad`` to measure absolute BetterHuman forward and
backward costs; gradient rows always use the differentiable Torch FK lane.
The fused lane requires BetterRobot's optional ``warp-lang>=1.8`` dependency.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import gc
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import torch
import better_robot as br

import better_human as bh
from better_human.bodies import MHRClassic
from better_human.core.pipeline import lbs, skinning_transforms_from_inverse

ROOT = Path(__file__).parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "MHR" / "converted" / "mhr_lod1.npz"
DEFAULT_REFERENCE_PATH = ROOT / "models" / "MHR" / "assets" / "mhr_model.pt"
CENTIMETERS_TO_METERS = 0.01
PARITY_ATOL_METERS = 5e-6


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batches", type=int, nargs="+", default=(1, 16, 256))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--grad",
        action="store_true",
        help="Measure absolute BetterHuman forward and backward costs",
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
    result = subprocess.run(
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()[0].strip()


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


def _inputs(
    batch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    identity = torch.linspace(-0.02, 0.02, 45, device=device).repeat(batch, 1)
    model_parameters = torch.linspace(-0.04, 0.04, 204, device=device).repeat(batch, 1)
    expression = torch.linspace(0.015, -0.015, 72, device=device).repeat(batch, 1)
    return identity, model_parameters, expression


def _betterhuman_inputs(
    neutral: bh.MHR,
    identity: torch.Tensor,
    model_parameters: torch.Tensor,
    expression: torch.Tensor,
) -> tuple[bh.MHR, torch.Tensor, dict[str, torch.Tensor]]:
    pose_body, scalar_q = neutral.from_classic(
        MHRClassic(identity[0], model_parameters[0], expression[0])
    )
    q = scalar_q.repeat(identity.shape[0], 1)
    shape_groups = {
        "identity": identity,
        "expression": expression,
        "proportions": model_parameters.index_select(
            -1,
            neutral._tree.proportion_parameter_indices,
        ),
    }
    return pose_body, q, shape_groups


def _vertices(
    body: bh.MHR,
    q: torch.Tensor,
    *,
    use_warp: bool,
) -> torch.Tensor:
    if not use_warp:
        return body.vertices(q)
    data = br.forward_kinematics(body.robot, q, use_warp=True)
    return body.vertices_from_data(data)


def _transform_and_lbs(
    body: bh.MHR,
    data: br.Data,
    corrective: torch.Tensor,
) -> torch.Tensor:
    native_poses = body.native_joint_poses(data)
    transforms = skinning_transforms_from_inverse(
        native_poses,
        body.values.inverse_bind_poses,
    )
    return lbs(
        body.values.v_shaped + corrective,
        transforms,
        body.values.skinning_weight_matrix,
        joint_scales=body.values.joint_scales,
        bind_joint_positions=body.values.bind_joint_positions,
    )


def _validate_vertices(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    batch: int,
    device: torch.device,
    path: str,
) -> float:
    expected_shape = (batch, 18439, 3)
    for source, vertices in (("TorchScript", expected), (path, actual)):
        if vertices.shape != expected_shape:
            raise RuntimeError(
                f"MHR {source} produced shape {tuple(vertices.shape)}; expected {expected_shape}"
            )
        if (
            vertices.dtype != torch.float32
            or vertices.device.type != device.type
            or not torch.isfinite(vertices).all()
        ):
            raise RuntimeError(f"MHR {source} produced invalid dtype, device, or non-finite values")
    error = (actual - expected).abs().max().item()
    if error > PARITY_ATOL_METERS:
        raise RuntimeError(
            f"MHR batch {batch} {path} max error {error:.9g} exceeds "
            f"{PARITY_ATOL_METERS:.1e}"
        )
    return error


def _agreement_results(
    reference_path: Path,
    model_path: Path,
    batches: tuple[int, ...],
    device: torch.device,
    use_warp: bool,
) -> list[tuple[int, bool, float, float]]:
    reference = torch.jit.load(reference_path, map_location=device)
    reference.eval()
    results: list[tuple[int, bool, float, float]] = []
    for use_correctives in (True, False):
        neutral = bh.MHR(
            model_path,
            lod=1,
            use_correctives=use_correctives,
            compute_mass=False,
            device=device,
        )
        for batch in batches:
            identity, model_parameters, expression = _inputs(batch, device)
            pose_body, q, shape_groups = _betterhuman_inputs(
                neutral,
                identity,
                model_parameters,
                expression,
            )
            expected = reference(
                identity,
                model_parameters,
                expression,
                use_correctives,
            )[0] * CENTIMETERS_TO_METERS
            streaming_vertices = _vertices(pose_body, q, use_warp=use_warp)
            full_vertices = neutral(q, **shape_groups)
            results.append(
                (
                    batch,
                    use_correctives,
                    _validate_vertices(
                        expected,
                        streaming_vertices,
                        batch=batch,
                        device=device,
                        path="streaming poses",
                    ),
                    _validate_vertices(
                        expected,
                        full_vertices,
                        batch=batch,
                        device=device,
                        path="full forward",
                    ),
                )
            )
            pose_body = q = shape_groups = None
            expected = streaming_vertices = full_vertices = None
            identity = model_parameters = expression = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        neutral = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    reference = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _reference_results(
    path: Path,
    batches: tuple[int, ...],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> list[tuple[int, bool, str, float]]:
    model = torch.jit.load(path, map_location=device)
    model.eval()
    results: list[tuple[int, bool, str, float]] = []
    for batch in batches:
        identity, model_parameters, expression = _inputs(batch, device)
        for use_correctives in (True, False):

            def forward() -> tuple[torch.Tensor, torch.Tensor]:
                return model(identity, model_parameters, expression, use_correctives)

            median = _median_milliseconds(
                forward,
                device=device,
                warmup=warmup,
                iterations=iterations,
            )
            results.append((batch, use_correctives, "TorchScript shape+pose", median))
    # Emptying the allocator cannot reclaim the last live model and inputs.
    forward = None
    identity = model_parameters = expression = None
    model = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _betterhuman_results(
    path: Path,
    batches: tuple[int, ...],
    device: torch.device,
    warmup: int,
    iterations: int,
    use_warp: bool,
) -> tuple[list[tuple[int, bool, str, float]], list[tuple[int, bool, str, float]]]:
    results: list[tuple[int, bool, str, float]] = []
    breakdown: list[tuple[int, bool, str, float]] = []
    for use_correctives in (True, False):
        neutral = bh.MHR(
            path,
            lod=1,
            use_correctives=use_correctives,
            compute_mass=False,
            device=device,
        )
        for batch in batches:
            identity, model_parameters, expression = _inputs(batch, device)
            pose_body, q, shape_groups = _betterhuman_inputs(
                neutral,
                identity,
                model_parameters,
                expression,
            )

            def pose_only() -> torch.Tensor:
                return _vertices(pose_body, q, use_warp=use_warp)

            def shape_and_pose() -> torch.Tensor:
                return neutral(q, **shape_groups)

            results.append(
                (
                    batch,
                    use_correctives,
                    "BetterHuman streaming poses" + (" (Warp requested)" if use_warp else ""),
                    _median_milliseconds(
                        pose_only,
                        device=device,
                        warmup=warmup,
                        iterations=iterations,
                    ),
                )
            )
            results.append(
                (
                    batch,
                    use_correctives,
                    "BetterHuman full shape+pose" + (" (Torch FK)" if use_warp else ""),
                    _median_milliseconds(
                        shape_and_pose,
                        device=device,
                        warmup=warmup,
                        iterations=iterations,
                    ),
                )
            )

            data = br.forward_kinematics(pose_body.robot, q, use_warp=use_warp)
            corrective = pose_body._pose_corrective_policy(pose_body, data)
            breakdown.extend(
                (
                    (
                        batch,
                        use_correctives,
                        "FK" + (" (Warp requested)" if use_warp else ""),
                        _median_milliseconds(
                            lambda: br.forward_kinematics(pose_body.robot, q, use_warp=use_warp),
                            device=device,
                            warmup=warmup,
                            iterations=iterations,
                        ),
                    ),
                    (
                        batch,
                        use_correctives,
                        "corrective",
                        _median_milliseconds(
                            lambda: pose_body._pose_corrective_policy(pose_body, data),
                            device=device,
                            warmup=warmup,
                            iterations=iterations,
                        ),
                    ),
                    (
                        batch,
                        use_correctives,
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
            # Bound peak memory by releasing each batch before allocator cleanup.
            pose_only = shape_and_pose = None
            pose_body = q = shape_groups = None
            identity = model_parameters = expression = None
            data = corrective = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        neutral = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results, breakdown


def _gradient_results(
    path: Path,
    batches: tuple[int, ...],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> list[tuple[int, bool, str, float, float, float]]:
    results: list[tuple[int, bool, str, float, float, float]] = []
    for use_correctives in (True, False):
        neutral = bh.MHR(
            path,
            lod=1,
            use_correctives=use_correctives,
            compute_mass=False,
            device=device,
        )
        for batch in batches:
            identity, model_parameters, expression = _inputs(batch, device)
            pose_body, q, shape_groups = _betterhuman_inputs(
                neutral,
                identity,
                model_parameters,
                expression,
            )
            full_q = q.detach().requires_grad_()
            full_shape_groups = {
                group: value.detach().requires_grad_()
                for group, value in shape_groups.items()
            }

            def full_forward() -> torch.Tensor:
                return neutral(full_q, **full_shape_groups)

            full_forward_ms, full_forward_backward_ms = _gradient_medians(
                full_forward,
                [full_q, *full_shape_groups.values()],
                device=device,
                warmup=warmup,
                iterations=iterations,
            )
            results.append(
                (
                    batch,
                    use_correctives,
                    "A — full shape+pose (q + shape groups)",
                    full_forward_ms,
                    full_forward_backward_ms,
                    full_forward_backward_ms / full_forward_ms,
                )
            )

            streaming_q = q.detach().requires_grad_()

            def streaming_forward() -> torch.Tensor:
                return pose_body.vertices(streaming_q)

            streaming_forward_ms, streaming_forward_backward_ms = _gradient_medians(
                streaming_forward,
                [streaming_q],
                device=device,
                warmup=warmup,
                iterations=iterations,
            )
            results.append(
                (
                    batch,
                    use_correctives,
                    "B — streaming poses (q)",
                    streaming_forward_ms,
                    streaming_forward_backward_ms,
                    streaming_forward_backward_ms / streaming_forward_ms,
                )
            )

            full_forward = streaming_forward = None
            pose_body = q = shape_groups = None
            full_q = full_shape_groups = streaming_q = None
            identity = model_parameters = expression = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        neutral = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def _print_workloads(
    reference_rows: list[tuple[int, bool, str, float]],
    betterhuman_rows: list[tuple[int, bool, str, float]],
) -> None:
    references = {
        (batch, correctives): median
        for batch, correctives, _, median in reference_rows
    }
    full_forwards = {
        (batch, correctives): (path, median)
        for batch, correctives, path, median in betterhuman_rows
        if path.startswith("BetterHuman full shape+pose")
    }
    streaming_poses = {
        (batch, correctives): (path, median)
        for batch, correctives, path, median in betterhuman_rows
        if path.startswith("BetterHuman streaming poses")
    }
    keys = sorted(references, key=lambda item: (item[0], not item[1]))

    print("\n## Workload A — full forward (new shape + new pose every call)\n")
    print("Both rows compute full shape and pose; each ratio compares identical work.")
    print("\n| Batch | Correctives | Path | Median ms | × reference |")
    print("|---:|:---:|---|---:|---:|")
    for batch, correctives in keys:
        reference = references[(batch, correctives)]
        path, median = full_forwards[(batch, correctives)]
        state = "on" if correctives else "off"
        print(f"| {batch} | {state} | TorchScript full shape+pose | {reference:.6f} | 1.000000 |")
        print(f"| {batch} | {state} | {path} | {median:.6f} | {median / reference:.6f} |")

    print("\n## Workload B — streaming poses (shape fixed, poses vary per call)\n")
    print(
        "BetterHuman bakes shape once outside timing. TorchScript has no "
        "shape-amortized API, so its per-frame cost remains a full shape+pose "
        "forward; ratios compare the same user workload, not matched FLOPs."
    )
    print("\n| Batch | Correctives | Per-frame path | Median ms | × reference |")
    print("|---:|:---:|---|---:|---:|")
    for batch, correctives in keys:
        reference = references[(batch, correctives)]
        path, median = streaming_poses[(batch, correctives)]
        state = "on" if correctives else "off"
        print(f"| {batch} | {state} | TorchScript full shape+pose | {reference:.6f} | 1.000000 |")
        print(f"| {batch} | {state} | {path} | {median:.6f} | {median / reference:.6f} |")


def _print_table(
    title: str,
    rows: list[tuple[int, bool, str, float]],
) -> None:
    print(f"\n## {title}\n")
    print("| Batch | Correctives | Path | Median ms |")
    print("|---:|:---:|---|---:|")
    for batch, correctives, path, median in rows:
        print(f"| {batch} | {'on' if correctives else 'off'} | {path} | {median:.6f} |")


def _print_agreement(rows: list[tuple[int, bool, float, float]]) -> None:
    print("\n## Pre-timing output agreement\n")
    print(f"Gate: max absolute vertex error ≤ {PARITY_ATOL_METERS:.1e} m.")
    print("\n| Batch | Correctives | Streaming poses max abs m | Full forward max abs m |")
    print("|---:|:---:|---:|---:|")
    for batch, correctives, streaming_error, full_error in rows:
        print(
            f"| {batch} | {'on' if correctives else 'off'} | "
            f"{streaming_error:.17g} | {full_error:.17g} |"
        )


def _print_gradients(rows: list[tuple[int, bool, str, float, float, float]]) -> None:
    print("\n## BetterHuman absolute gradient performance\n")
    print(
        "Workload A differentiates q and all shape groups; workload B differentiates "
        "q with shape baked outside timing. Both use Torch FK and Torch autograd, with "
        "no upstream row."
    )
    print("F+B / F is the training-step multiplier, not a backward-only median.")
    print(
        "\n| Batch | Correctives | Workload | Forward-only ms | "
        "Forward + backward ms | F+B / F |"
    )
    print("|---:|:---:|---|---:|---:|---:|")
    for batch, correctives, workload, forward, forward_backward, ratio in rows:
        print(
            f"| {batch} | {'on' if correctives else 'off'} | {workload} | "
            f"{forward:.6f} | {forward_backward:.6f} | {ratio:.6f} |"
        )


def main() -> None:
    """Run the requested benchmark grid and print copyable Markdown tables."""
    arguments = _arguments()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if not arguments.model_path.is_file():
        raise FileNotFoundError(f"converted MHR archive is absent: {arguments.model_path}")
    if not arguments.reference_path.is_file():
        raise FileNotFoundError(f"MHR TorchScript reference is absent: {arguments.reference_path}")
    batches = tuple(arguments.batches)
    print(f"torch={torch.__version__}")
    print(f"better_robot={br.__version__}")
    print(
        f"device={device}"
        + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
    )
    if device.type == "cuda":
        print(f"cuda_runtime={torch.version.cuda}")
        print(f"cuda_driver={_cuda_driver_version()}")
    print(f"cpu_threads={torch.get_num_threads()}")
    print(f"cpu_interop_threads={torch.get_num_interop_threads()}")
    print(f"warmup={arguments.warmup}; iterations={arguments.iterations}; batches={batches}")
    print(f"use_warp={arguments.use_warp}")
    print(f"grad={arguments.grad}")
    print("timing_caveat=shared-host scheduling and concurrent load may affect observational medians")
    print(f"model_path={arguments.model_path}")
    print(f"reference_path={arguments.reference_path}")

    with torch.inference_mode():
        agreement = _agreement_results(
            arguments.reference_path,
            arguments.model_path,
            batches,
            device,
            arguments.use_warp,
        )
        _print_agreement(agreement)
        reference = _reference_results(
            arguments.reference_path,
            batches,
            device,
            arguments.warmup,
            arguments.iterations,
        )
        betterhuman, breakdown = _betterhuman_results(
            arguments.model_path,
            batches,
            device,
            arguments.warmup,
            arguments.iterations,
            arguments.use_warp,
        )
    _print_workloads(reference, betterhuman)
    _print_table("BetterHuman streaming-pose breakdown", breakdown)
    if arguments.grad:
        gradients = _gradient_results(
            arguments.model_path,
            batches,
            device,
            arguments.warmup,
            arguments.iterations,
        )
        _print_gradients(gradients)


if __name__ == "__main__":
    main()
