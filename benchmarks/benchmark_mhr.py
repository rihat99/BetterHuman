"""Benchmark BetterHuman MHR against the released LOD1 TorchScript model.

The reference recomputes shape on every call. BetterHuman reports both its
shape-baked pose path and an apples-to-apples shape-plus-pose path.

Run the pinned protocol on CPU with::

    uv run python benchmarks/benchmark_mhr.py --device cpu

Use ``--device cuda --use-warp`` to exercise BetterRobot's fused FK lane for
the pose-only rows. Batched shape values currently keep the fair shape+pose
rows on the Torch FK lane. The fused lane requires BetterRobot's optional
``warp-lang>=1.8`` dependency.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import gc
from pathlib import Path
import statistics
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


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
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
    return arguments


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


def _inputs(
    batch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    identity = torch.linspace(-0.02, 0.02, 45, device=device).repeat(batch, 1)
    model_parameters = torch.linspace(-0.04, 0.04, 204, device=device).repeat(batch, 1)
    expression = torch.linspace(0.015, -0.015, 72, device=device).repeat(batch, 1)
    return identity, model_parameters, expression


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
        body.values.skinning_weights,
        body.structure.skinning_indices,
        joint_scales=body.values.joint_scales,
        bind_joint_positions=body.values.bind_joint_positions,
    )


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
            pose_body, scalar_q = neutral.from_classic(MHRClassic(identity[0], model_parameters[0], expression[0]))
            q = scalar_q.repeat(batch, 1)
            shape_groups = {
                "identity": identity,
                "expression": expression,
                "proportions": model_parameters.index_select(-1, neutral._proportion_parameter_indices),
            }

            def pose_only() -> torch.Tensor:
                return _vertices(pose_body, q, use_warp=use_warp)

            def shape_and_pose() -> torch.Tensor:
                current = neutral.with_shape(**shape_groups)
                return current.vertices(q)

            results.append(
                (
                    batch,
                    use_correctives,
                    "BetterHuman pose-only" + (" (Warp requested)" if use_warp else ""),
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
                    "BetterHuman shape+pose" + (" (Torch FK)" if use_warp else ""),
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
            pose_body = scalar_q = q = shape_groups = None
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


def _print_table(
    title: str,
    rows: list[tuple[int, bool, str, float]],
) -> None:
    print(f"\n## {title}\n")
    print("| Batch | Correctives | Path | Median ms |")
    print("|---:|:---:|---|---:|")
    for batch, correctives, path, median in rows:
        print(f"| {batch} | {'on' if correctives else 'off'} | {path} | {median:.6f} |")


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
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"cpu_threads={torch.get_num_threads()}")
    print(f"cpu_interop_threads={torch.get_num_interop_threads()}")
    print(f"warmup={arguments.warmup}; iterations={arguments.iterations}; batches={batches}")
    print(f"use_warp={arguments.use_warp}")

    with torch.inference_mode():
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
    _print_table("End-to-end", sorted((*reference, *betterhuman), key=lambda row: (row[0], not row[1], row[2])))
    _print_table("BetterHuman pose-only breakdown", breakdown)


if __name__ == "__main__":
    main()
