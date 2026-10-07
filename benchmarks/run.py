"""Benchmark the BetterHuman core (bake, vertices, fk, rnea) against upstream smplx and the MHR TorchScript model.

Usage: ``uv run python benchmarks/run.py --device cpu --threads 8 --batches 1 64 1024 --out results/x.json``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix

import better_human as bh
from better_robot.dynamics.state import inverse_dynamics
from better_robot.kinematics import forward_kinematics_state

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PACKAGE_ROOT = ROOT / "references" / "parametric_body" / "smplx"
SMPL_PATH = ROOT / "models" / "smpl" / "SMPL_NEUTRAL.npz"
SMPLX_PATH = ROOT / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
MHR_PATH = ROOT / "models" / "MHR" / "converted" / "mhr_lod1.npz"
MHR_REFERENCE_PATH = ROOT / "models" / "MHR" / "assets" / "mhr_model.pt"
FAMILIES = ("SMPL", "SMPL-X", "MHR")
# Peak bytes per vertex per batch element, used only to skip oversized cells: measured peak RSS of our CPU
# forward / forward+backward at batch 512 was at most 128 / 156 B per vertex (MHR LOD1); ~1.5x margin.
FORWARD_BYTES_PER_VERTEX = 4 * 48
BACKWARD_BYTES_PER_VERTEX = 4 * 64
TIME_BUDGET_SECONDS = 3.0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 64, 1024, 8192])
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--iterations", type=int, default=30, help="upper bound on timed iterations per cell")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--memory-gb", type=float, default=8.0, help="skip cells estimated above this")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() or "unknown"


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _timed(function: Callable[[], Any], device: torch.device, iterations: int) -> dict[str, float | int]:
    """Median and p90 wall time (ms) of ``function`` after warm-up; iteration count adapts to a time budget."""

    def once() -> float:
        started = time.perf_counter_ns()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return (time.perf_counter_ns() - started) * 1e-6

    warmup = [once() for _ in range(3)]
    count = max(5, min(iterations, int(TIME_BUDGET_SECONDS * 1e3 / max(min(warmup), 1e-3))))
    durations = sorted(once() for _ in range(count))
    p90 = durations[min(len(durations) - 1, math.ceil(0.9 * len(durations)) - 1)]
    return {"median_ms": statistics.median(durations), "p90_ms": p90, "iterations": count}


def _vendored_smplx() -> tuple[Any, type[Any]]:
    """Import the vendored upstream ``smplx`` package and its ``Struct`` type."""
    sys.path.insert(0, str(REFERENCE_PACKAGE_ROOT))
    package = importlib.import_module("smplx")
    if REFERENCE_PACKAGE_ROOT.resolve() not in Path(package.__file__).resolve().parents:
        raise RuntimeError(f"expected vendored smplx under {REFERENCE_PACKAGE_ROOT}, got {package.__file__}")
    return package, importlib.import_module("smplx.utils").Struct


def _smpl_data_struct(path: Path, struct_type: type[Any]) -> Any:
    """Converted SMPL archive in upstream's representation (sparse ``J_regressor``)."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if "J_regressor" not in arrays:
        shape = tuple(int(value) for value in arrays.pop("J_regressor_shape"))
        arrays["J_regressor"] = csr_matrix(
            (arrays.pop("J_regressor_data"), arrays.pop("J_regressor_indices"), arrays.pop("J_regressor_indptr")),
            shape=shape,
        )
    return struct_type(**arrays)


def _upstream_case(family: str, batch: int, device: torch.device) -> tuple[Callable[[bool], Any], int]:
    """``(run(grad) -> vertices or grads, num_vertices)`` for upstream smplx with betas, expression and pose."""
    package, struct_type = _vendored_smplx()
    common = {
        "dtype": torch.float32,
        "create_betas": False,
        "create_global_orient": False,
        "create_body_pose": False,
        "create_transl": False,
        "batch_size": batch,
    }
    generator = torch.Generator().manual_seed(0)

    def rand(width: int, scale: float) -> torch.Tensor:
        return (scale * torch.randn(batch, width, generator=generator)).to(device)

    inputs = {
        "betas": rand(10, 1.0),
        "global_orient": rand(3, 0.2),
        "body_pose": rand(63 if family != "SMPL" else 69, 0.2),
    }
    inputs["transl"] = rand(3, 0.1)
    if family == "SMPL":
        model = package.SMPL(
            model_path=str(SMPL_PATH), data_struct=_smpl_data_struct(SMPL_PATH, struct_type), num_betas=10, **common
        )
        pose_names = ("global_orient", "body_pose")
    else:
        model = package.SMPLX(
            model_path=str(SMPLX_PATH),
            num_betas=10,
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
            **common,
        )
        inputs |= {"left_hand_pose": rand(45, 0.2), "right_hand_pose": rand(45, 0.2), "expression": rand(10, 0.5)}
        inputs |= {"jaw_pose": rand(3, 0.1), "leye_pose": rand(3, 0.1), "reye_pose": rand(3, 0.1)}
        pose_names = ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose")
    model = model.eval().to(device)

    def run(grad: bool) -> Any:
        if not grad:
            with torch.no_grad():
                return model(**inputs, return_verts=True).vertices
        leaves = {name: inputs[name].clone().requires_grad_(True) for name in pose_names}
        vertices = model(**(inputs | leaves), return_verts=True).vertices
        return torch.autograd.grad(vertices.sum(), list(leaves.values()))

    return run, model.v_template.shape[0]


def _spec(family: str, device: torch.device) -> Any:
    if family == "SMPL":
        return bh.families.smpl(SMPL_PATH, num_betas=10, device=device)
    if family == "SMPL-X":
        return bh.families.smplx(SMPLX_PATH, num_betas=10, num_expression=10, hands=True, face=True, device=device)
    return bh.families.mhr(MHR_PATH, lod=1, expression=True, correctives=True, device=device)


def _identity_inputs(body: Any, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return {
        name: (0.5 * torch.randn(width, generator=generator)).to(device) for name, width in body.identity_groups.items()
    }


def _stream_inputs(body: Any, batch: int, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(2)
    return {
        name: (0.3 * torch.randn(batch, width, generator=generator)).to(device)
        for name, width in body.stream_groups.items()
    }


def _baselines(
    family: str,
    batch: int,
    device: torch.device,
    iterations: int,
    skip: Callable[[str, str, int], bool],
    rows: list[dict[str, Any]],
    reference_model: Any,
) -> dict[str, float]:
    """Upstream smplx (SMPL, SMPL-X) or TorchScript (MHR) medians at this batch; rows appended in place."""
    baselines: dict[str, float] = {}
    if family in ("SMPL", "SMPL-X"):
        forward_skipped = skip("vertices", "upstream", FORWARD_BYTES_PER_VERTEX)
        backward_skipped = skip("vertices_backward", "upstream", BACKWARD_BYTES_PER_VERTEX)
        if forward_skipped:
            return baselines
        run, _ = _upstream_case(family, batch, device)
        timing = _timed(lambda: run(False), device, iterations)
        baselines["forward"] = timing["median_ms"]
        rows.append({"family": family, "op": "vertices", "batch": batch, "variant": "upstream"} | timing)
        if not backward_skipped:
            timing = _timed(lambda: run(True), device, iterations)
            baselines["backward"] = timing["median_ms"]
            rows.append({"family": family, "op": "vertices_backward", "batch": batch, "variant": "upstream"} | timing)
        return baselines
    if skip("vertices", "torchscript", FORWARD_BYTES_PER_VERTEX):
        return baselines
    mhr_inputs = (
        torch.linspace(-0.02, 0.02, 45, device=device).repeat(batch, 1),
        torch.linspace(-0.04, 0.04, 204, device=device).repeat(batch, 1),
        torch.linspace(0.015, -0.015, 72, device=device).repeat(batch, 1),
    )

    def reference_forward() -> torch.Tensor:
        with torch.no_grad():
            return reference_model(*mhr_inputs, True)[0]

    timing = _timed(reference_forward, device, iterations)
    baselines["forward"] = timing["median_ms"]
    rows.append({"family": family, "op": "vertices", "batch": batch, "variant": "torchscript"} | timing)
    return baselines


def _bench_family(family: str, arguments: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    """All rows (ours and baseline) for one family."""
    rows: list[dict[str, Any]] = []
    limit = arguments.memory_gb * 1e9
    spec = _spec(family, device)
    bodies = {"mass": bh.Body(spec, device=device, mass=True), "no_mass": bh.Body(spec, device=device, mass=False)}
    body = bodies["mass"]
    identity_inputs = _identity_inputs(body, device)
    for variant, candidate in bodies.items():
        timing = _timed(lambda candidate=candidate: candidate.bake(**identity_inputs), device, arguments.iterations)
        rows.append({"family": family, "op": "bake", "batch": 1, "variant": variant} | timing)
    identity = body.bake(**identity_inputs)
    num_vertices = body.num_vertices
    vertex_bodies: dict[str, tuple[Any, bool]] = {"default": (identity, family == "MHR")}
    if family == "SMPL-X":
        frozen = bh.Body(spec, device=device, mass=False, stream_joints=False).bake(**identity_inputs)
        vertex_bodies["expression_stream_joints"] = (identity, True)
        vertex_bodies["expression_frozen_joints"] = (frozen, True)
    reference_model = torch.jit.load(str(MHR_REFERENCE_PATH), map_location=device) if family == "MHR" else None

    for batch in arguments.batches:

        def skip(op: str, variant: str, bytes_per_vertex: int) -> bool:
            estimate = batch * num_vertices * bytes_per_vertex
            if estimate <= limit:
                return False
            reason = f"estimated {estimate / 1e9:.1f} GB > {arguments.memory_gb:g} GB"
            rows.append({"family": family, "op": op, "batch": batch, "variant": variant, "skipped": reason})
            return True

        q = identity.q_neutral.expand(batch, identity.nq).clone()
        baselines = _baselines(family, batch, device, arguments.iterations, skip, rows, reference_model)

        def add(op: str, variant: str, function: Callable[[], Any], baseline: str | None) -> None:
            timing = _timed(function, device, arguments.iterations)
            row = {"family": family, "op": op, "batch": batch, "variant": variant} | timing
            if baseline in baselines:
                row["baseline_median_ms"] = baselines[baseline]
                row["ratio_vs_baseline"] = timing["median_ms"] / baselines[baseline]
            rows.append(row)

        for variant, (vertex_identity, with_stream) in vertex_bodies.items():
            stream = _stream_inputs(body, batch, device) if with_stream else None
            if not skip("vertices", variant, FORWARD_BYTES_PER_VERTEX):
                with torch.no_grad():
                    add("vertices", variant, lambda i=vertex_identity, s=stream: i.forward(q, s).vertices(), "forward")
            if not skip("vertices_backward", variant, BACKWARD_BYTES_PER_VERTEX):
                leaf = q.clone().requires_grad_(True)

                def backward(i: Any = vertex_identity, s: Any = stream) -> Any:
                    return torch.autograd.grad(i.forward(leaf, s).vertices().sum(), leaf)

                add("vertices_backward", variant, backward, "backward")

        model = identity.model
        with torch.no_grad():
            add("fk", "default", lambda: forward_kinematics_state(model, q), None)
            state = forward_kinematics_state(model, q)
            generator = torch.Generator().manual_seed(3)
            v = (0.1 * torch.randn(batch, model.nv, generator=generator)).to(device)
            a = (0.1 * torch.randn(batch, model.nv, generator=generator)).to(device)
            add("rnea", "default", lambda: inverse_dynamics(model, state, v, a), None)
        print(f"{family} batch {batch} done", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    arguments = _arguments()
    if arguments.threads is not None:
        torch.set_num_threads(arguments.threads)
    device = torch.device(arguments.device)
    torch.manual_seed(0)
    rows: list[dict[str, Any]] = []
    for family in arguments.families:
        rows.extend(_bench_family(family, arguments, device))
    result = {
        "torch": torch.__version__,
        "device": arguments.device,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else _cpu_name(),
        "threads": torch.get_num_threads(),
        "git_commit": _git_commit(),
        "dtype": "float32",
        "memory_limit_gb": arguments.memory_gb,
        "rows": rows,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
