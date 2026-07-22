"""Run the compact CUDA benchmark for all BetterHuman body families.

The benchmark emits two tables from one invocation:

* fixed-shape vertex streaming against each released reference forward;
* absolute BetterRobot FK and RNEA timings, with no RNEA reference.

Only the eager Torch and fused Warp lanes are included. Shape construction,
mass integration, and Warp compilation are outside timing. FK and RNEA reuse
preallocated workspaces; vertex timings use the public end-to-end APIs.

Run the recorded protocol with::

    CUDA_VISIBLE_DEVICES=7 uv run python benchmarks/benchmark_bodies.py
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import gc
import importlib
from importlib.metadata import version
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any
import warnings

import better_robot as br
import numpy as np
from scipy.sparse import csr_matrix
import torch

import better_human as bh
from better_human.bodies import MHRClassic, SMPLClassic, SMPLXClassic
from better_human.core import ParametricBody

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PACKAGE_ROOT = ROOT / "references" / "parametric_body" / "smplx"
DEFAULT_SMPL_PATH = ROOT / "models" / "smpl" / "SMPL_NEUTRAL.npz"
DEFAULT_SMPLX_PATH = ROOT / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
DEFAULT_MHR_PATH = ROOT / "models" / "MHR" / "converted" / "mhr_lod1.npz"
DEFAULT_MHR_REFERENCE_PATH = ROOT / "models" / "MHR" / "assets" / "mhr_model.pt"
CENTIMETERS_TO_METERS = 0.01
SMPL_VERTEX_ATOL_METERS = 1e-6
MHR_VERTEX_ATOL_METERS = 5e-6
FK_RTOL = 2e-5
FK_ATOL = 2e-6
RNEA_RTOL = 3e-4
RNEA_ATOL = 2e-4


@dataclass(frozen=True)
class PreparedCase:
    """One fixed-shape body and its matching released-reference callback."""

    #: Body-family label printed in tables.
    name: str
    #: Shape-baked body with mass properties for FK, vertices, and RNEA.
    body: ParametricBody
    #: Reference-matched pose batch with shape ``(batch, nq)``.
    q: torch.Tensor
    #: Full-forward callback for upstream ``smplx`` or MHR TorchScript.
    reference_forward: Callable[[], torch.Tensor]
    #: Expected vertex count for the selected body fixture.
    vertex_count: int
    #: Maximum permitted reference vertex error in meters.
    vertex_atol: float


@dataclass(frozen=True)
class ForwardRow:
    """One row in the fixed-shape vertex-streaming table."""

    #: Body-family label.
    name: str
    #: Pose batch size.
    batch: int
    #: Released-reference full-forward median in milliseconds.
    reference_ms: float
    #: BetterHuman eager-Torch median in milliseconds.
    torch_ms: float
    #: BetterHuman Warp median in milliseconds.
    warp_ms: float
    #: Worst reference error across the Torch and Warp paths.
    vertex_error: float


@dataclass(frozen=True)
class DynamicsRow:
    """One row in the absolute BetterRobot FK/RNEA table."""

    #: Body-family label.
    name: str
    #: Number of joints in the BetterRobot model, including universe.
    joints: int
    #: Generalized velocity dimension.
    nv: int
    #: Query batch size.
    batch: int
    #: Eager-Torch FK median in milliseconds.
    fk_torch_ms: float
    #: Warp FK median in milliseconds.
    fk_warp_ms: float
    #: Maximum sign-aligned Torch-versus-Warp FK pose difference.
    fk_error: float
    #: Eager-Torch RNEA median in milliseconds.
    rnea_torch_ms: float
    #: Warp RNEA median in milliseconds.
    rnea_warp_ms: float
    #: Maximum absolute Torch-versus-Warp torque difference.
    rnea_error: float


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smpl-path", type=Path, default=DEFAULT_SMPL_PATH)
    parser.add_argument("--smplx-path", type=Path, default=DEFAULT_SMPLX_PATH)
    parser.add_argument("--mhr-path", type=Path, default=DEFAULT_MHR_PATH)
    parser.add_argument(
        "--mhr-reference-path",
        type=Path,
        default=DEFAULT_MHR_REFERENCE_PATH,
    )
    parser.add_argument("--batches", type=int, nargs="+", default=(1, 256))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    arguments = parser.parse_args()
    if any(batch < 1 for batch in arguments.batches):
        parser.error("--batches values must be positive")
    if arguments.warmup < 0 or arguments.iterations < 1:
        parser.error("--warmup must be nonnegative and --iterations positive")
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


def _git_state(repository: Path) -> tuple[str, bool]:
    """Return one repository's full revision and dirty-worktree flag."""
    revision = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ("git", "-C", str(repository), "status", "--porcelain"),
        check=True,
        capture_output=True,
        text=True,
    )
    return revision.stdout.strip(), bool(status.stdout.strip())


def _synchronize() -> None:
    torch.cuda.synchronize()


def _median_milliseconds(
    callback: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        callback()
    _synchronize()
    durations: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        callback()
        _synchronize()
        durations.append((time.perf_counter_ns() - started) * 1e-6)
    return statistics.median(durations)


def _vendored_smplx() -> tuple[Any, type[Any]]:
    """Import the vendored upstream package and its data-structure type."""
    sys.path.insert(0, str(REFERENCE_PACKAGE_ROOT))
    package = importlib.import_module("smplx")
    package_path = Path(package.__file__).resolve()
    if REFERENCE_PACKAGE_ROOT.resolve() not in package_path.parents:
        raise RuntimeError(f"expected vendored smplx under {REFERENCE_PACKAGE_ROOT}, got {package_path}")
    struct_type = importlib.import_module("smplx.utils").Struct
    return package, struct_type


def _smpl_data_struct(path: Path, struct_type: type[Any]) -> Any:
    """Load the converted SMPL archive in upstream's expected representation."""
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


def _reference_smpl_model(
    name: str,
    path: Path,
    package: Any,
    struct_type: type[Any],
    device: torch.device,
    batch: int,
) -> torch.nn.Module:
    """Construct an external-parameter upstream SMPL-family model."""
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


def _smpl_inputs(
    batch: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], SMPLClassic]:
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
    return reference, {"betas": betas}, classic


def _smplx_inputs(
    batch: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], SMPLXClassic]:
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
    shape_groups = {"betas": betas, "expression": expression}
    return reference, shape_groups, classic


def _prepare_smpl_case(
    name: str,
    path: Path,
    neutral: ParametricBody,
    package: Any,
    struct_type: type[Any],
    batch: int,
    device: torch.device,
) -> PreparedCase:
    """Prepare one SMPL-family batch with fixed shape and varying poses."""
    reference_model = _reference_smpl_model(
        name,
        path,
        package,
        struct_type,
        device,
        batch,
    )
    if name == "SMPL":
        reference_inputs, shape_groups, classic = _smpl_inputs(batch, device)
        vertex_count = 6890
    else:
        reference_inputs, shape_groups, classic = _smplx_inputs(batch, device)
        vertex_count = 10475
    scalar_shape = {group: value[0] for group, value in shape_groups.items()}
    body = neutral.with_shape(**scalar_shape)
    q = body.from_classic(classic).contiguous()

    def reference_forward() -> torch.Tensor:
        return reference_model(**reference_inputs).vertices

    return PreparedCase(
        name=name,
        body=body,
        q=q,
        reference_forward=reference_forward,
        vertex_count=vertex_count,
        vertex_atol=SMPL_VERTEX_ATOL_METERS,
    )


def _prepare_mhr_case(
    neutral: bh.MHR,
    reference_model: torch.jit.ScriptModule,
    batch: int,
    device: torch.device,
) -> PreparedCase:
    """Prepare one MHR LOD1 batch with fixed shape and correctives enabled."""
    identity = torch.linspace(-0.02, 0.02, 45, device=device).repeat(batch, 1)
    model_parameters = torch.linspace(-0.04, 0.04, 204, device=device).repeat(batch, 1)
    expression = torch.linspace(0.015, -0.015, 72, device=device).repeat(batch, 1)
    body, scalar_q = neutral.from_classic(
        MHRClassic(identity[0], model_parameters[0], expression[0])
    )
    q = scalar_q.repeat(batch, 1).contiguous()

    def reference_forward() -> torch.Tensor:
        return (
            reference_model(identity, model_parameters, expression, True)[0]
            * CENTIMETERS_TO_METERS
        )

    return PreparedCase(
        name="MHR",
        body=body,
        q=q,
        reference_forward=reference_forward,
        vertex_count=18439,
        vertex_atol=MHR_VERTEX_ATOL_METERS,
    )


def _warp_vertices(case: PreparedCase) -> torch.Tensor:
    data = br.forward_kinematics(case.body.robot, case.q, use_warp=True)
    return case.body.vertices_from_data(data)


def _validate_vertices(
    case: PreparedCase,
    expected: torch.Tensor,
    torch_vertices: torch.Tensor,
    warp_vertices: torch.Tensor,
) -> float:
    expected_shape = (case.q.shape[0], case.vertex_count, 3)
    for source, vertices in (
        ("reference", expected),
        ("BetterHuman Torch", torch_vertices),
        ("BetterHuman Warp", warp_vertices),
    ):
        if vertices.shape != expected_shape:
            raise RuntimeError(
                f"{case.name} {source} produced {tuple(vertices.shape)}; expected {expected_shape}"
            )
        if vertices.dtype != torch.float32 or not vertices.is_cuda or not torch.isfinite(vertices).all():
            raise RuntimeError(f"{case.name} {source} produced invalid vertices")
    error = max(
        (torch_vertices - expected).abs().max().item(),
        (warp_vertices - expected).abs().max().item(),
    )
    if error > case.vertex_atol:
        raise RuntimeError(
            f"{case.name} batch {case.q.shape[0]} vertex error {error:.9g} "
            f"exceeds {case.vertex_atol:.1e} m"
        )
    return error


def _dynamics_inputs(
    model: br.Model,
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row_scale = torch.ones((batch, 1), device="cuda", dtype=torch.float32)
    if batch > 1:
        row_scale = torch.linspace(0.8, 1.2, batch, device="cuda").unsqueeze(-1)
    tangent = row_scale * torch.linspace(-0.03, 0.03, model.nv, device="cuda").unsqueeze(0)
    q_neutral = model.q_neutral.unsqueeze(0).repeat(batch, 1)
    q = model.integrate(q_neutral, tangent).contiguous()
    velocity = (
        row_scale * torch.linspace(-0.2, 0.2, model.nv, device="cuda").unsqueeze(0)
    ).contiguous()
    acceleration = (
        row_scale * torch.linspace(0.3, -0.3, model.nv, device="cuda").unsqueeze(0)
    ).contiguous()
    return q, velocity, acceleration


def _sign_aligned(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Align pose quaternion signs in ``actual`` to ``expected``."""
    dot = (actual[..., 3:] * expected[..., 3:]).sum(dim=-1, keepdim=True)
    sign = torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))
    return torch.cat((actual[..., :3], actual[..., 3:] * sign), dim=-1)


def _validate_fk(torch_data: br.Data, warp_data: br.Data) -> float:
    """Check Torch and Warp joint poses and return their worst difference."""
    max_error = 0.0
    for field in ("joint_pose_world", "joint_pose_local"):
        expected = getattr(torch_data, field)
        actual = getattr(warp_data, field)
        if expected is None or actual is None:
            raise RuntimeError(f"FK did not populate {field}")
        aligned = _sign_aligned(actual, expected)
        torch.testing.assert_close(
            aligned,
            expected,
            rtol=FK_RTOL,
            atol=FK_ATOL,
        )
        max_error = max(max_error, (aligned - expected).abs().max().item())
    return max_error


def _measure_case(
    case: PreparedCase,
    *,
    warmup: int,
    iterations: int,
) -> tuple[ForwardRow, DynamicsRow]:
    batch = case.q.shape[0]
    model = case.body.robot

    expected = case.reference_forward()
    torch_vertices = case.body.vertices(case.q)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warp_vertices = _warp_vertices(case)
    vertex_error = _validate_vertices(case, expected, torch_vertices, warp_vertices)
    expected = torch_vertices = warp_vertices = None

    q, velocity, acceleration = _dynamics_inputs(model, batch)
    fk_torch_data = model.create_data(batch_shape=(batch,), device=q.device, dtype=q.dtype)
    fk_warp_data = model.create_data(batch_shape=(batch,), device=q.device, dtype=q.dtype)
    fk_torch_data.q = q
    fk_warp_data.q = q
    rnea_torch_data = model.create_data(batch_shape=(batch,), device=q.device, dtype=q.dtype)
    rnea_warp_data = model.create_data(batch_shape=(batch,), device=q.device, dtype=q.dtype)

    torch_fk = br.forward_kinematics(model, fk_torch_data)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warp_fk = br.forward_kinematics(model, fk_warp_data, use_warp=True)
    fk_error = _validate_fk(torch_fk, warp_fk)

    torch_tau = br.rnea(
        model,
        q,
        velocity,
        acceleration,
        data=rnea_torch_data,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warp_tau = br.rnea(
            model,
            q,
            velocity,
            acceleration,
            data=rnea_warp_data,
            use_warp=True,
        )
    expected_tau_shape = (batch, model.nv)
    if torch_tau.shape != expected_tau_shape or warp_tau.shape != expected_tau_shape:
        raise RuntimeError(
            f"{case.name} RNEA produced {tuple(torch_tau.shape)} and "
            f"{tuple(warp_tau.shape)}; expected {expected_tau_shape}"
        )
    if not torch.isfinite(torch_tau).all() or not torch.isfinite(warp_tau).all():
        raise RuntimeError(f"{case.name} RNEA produced non-finite torque")
    torch.testing.assert_close(warp_tau, torch_tau, rtol=RNEA_RTOL, atol=RNEA_ATOL)
    rnea_error = (warp_tau - torch_tau).abs().max().item()

    reference_ms = _median_milliseconds(
        case.reference_forward,
        warmup=warmup,
        iterations=iterations,
    )
    torch_vertices_ms = _median_milliseconds(
        lambda: case.body.vertices(case.q),
        warmup=warmup,
        iterations=iterations,
    )
    warp_vertices_ms = _median_milliseconds(
        lambda: _warp_vertices(case),
        warmup=warmup,
        iterations=iterations,
    )
    fk_torch_ms = _median_milliseconds(
        lambda: br.forward_kinematics(model, fk_torch_data),
        warmup=warmup,
        iterations=iterations,
    )
    fk_warp_ms = _median_milliseconds(
        lambda: br.forward_kinematics(model, fk_warp_data, use_warp=True),
        warmup=warmup,
        iterations=iterations,
    )
    rnea_torch_ms = _median_milliseconds(
        lambda: br.rnea(model, q, velocity, acceleration, data=rnea_torch_data),
        warmup=warmup,
        iterations=iterations,
    )
    rnea_warp_ms = _median_milliseconds(
        lambda: br.rnea(
            model,
            q,
            velocity,
            acceleration,
            data=rnea_warp_data,
            use_warp=True,
        ),
        warmup=warmup,
        iterations=iterations,
    )

    return (
        ForwardRow(
            name=case.name,
            batch=batch,
            reference_ms=reference_ms,
            torch_ms=torch_vertices_ms,
            warp_ms=warp_vertices_ms,
            vertex_error=vertex_error,
        ),
        DynamicsRow(
            name=case.name,
            joints=model.njoints,
            nv=model.nv,
            batch=batch,
            fk_torch_ms=fk_torch_ms,
            fk_warp_ms=fk_warp_ms,
            fk_error=fk_error,
            rnea_torch_ms=rnea_torch_ms,
            rnea_warp_ms=rnea_warp_ms,
            rnea_error=rnea_error,
        ),
    )


def _release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _print_results(
    forward_rows: list[ForwardRow],
    dynamics_rows: list[DynamicsRow],
) -> None:
    print("\n## Fixed-shape vertex streaming\n")
    print(
        "Reference is a full per-frame forward (upstream smplx or MHR TorchScript); "
        "BetterHuman bakes shape outside timing. Ratios are workload-level, not matched FLOPs."
    )
    print("\n| Body | Batch | Reference full ms | BH Torch ms | × ref | BH Warp ms | × ref |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in forward_rows:
        print(
            f"| {row.name} | {row.batch} | {row.reference_ms:.6f} | "
            f"{row.torch_ms:.6f} | {row.torch_ms / row.reference_ms:.6f} | "
            f"{row.warp_ms:.6f} | {row.warp_ms / row.reference_ms:.6f} |"
        )

    print("\n## BetterRobot FK and RNEA\n")
    print(
        "Absolute CUDA medians with shape-specific mass properties and reused Data workspaces; "
        "RNEA has no original/reference comparison."
    )
    print(
        "\n| Body | Joints | nv | Batch | FK Torch ms | FK Warp ms | "
        "RNEA Torch ms | RNEA Warp ms |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in dynamics_rows:
        print(
            f"| {row.name} | {row.joints} | {row.nv} | {row.batch} | "
            f"{row.fk_torch_ms:.6f} | {row.fk_warp_ms:.6f} | "
            f"{row.rnea_torch_ms:.6f} | {row.rnea_warp_ms:.6f} |"
        )

    vertex_errors = ", ".join(
        f"{row.name} b{row.batch}={row.vertex_error:.9g}" for row in forward_rows
    )
    rnea_errors = ", ".join(
        f"{row.name} b{row.batch}={row.rnea_error:.9g}" for row in dynamics_rows
    )
    fk_errors = ", ".join(
        f"{row.name} b{row.batch}={row.fk_error:.9g}" for row in dynamics_rows
    )
    print(f"\nvertex_reference_max_abs_m: {vertex_errors}")
    print(f"fk_torch_warp_max_abs: {fk_errors}")
    print(f"rnea_torch_warp_max_abs: {rnea_errors}")
    print(
        f"gates: SMPL-family vertices ≤ {SMPL_VERTEX_ATOL_METERS:.1e} m; "
        f"MHR vertices ≤ {MHR_VERTEX_ATOL_METERS:.1e} m; "
        f"FK rtol={FK_RTOL:.1e}, atol={FK_ATOL:.1e}; "
        f"RNEA rtol={RNEA_RTOL:.1e}, atol={RNEA_ATOL:.1e}"
    )


def _measure_batches(
    prepare_case: Callable[[int], PreparedCase],
    batches: tuple[int, ...],
    *,
    warmup: int,
    iterations: int,
) -> tuple[list[ForwardRow], list[DynamicsRow]]:
    """Measure every requested batch for one already-loaded body family."""
    forward_rows: list[ForwardRow] = []
    dynamics_rows: list[DynamicsRow] = []
    for batch in batches:
        case = prepare_case(batch)
        forward_row, dynamics_row = _measure_case(
            case,
            warmup=warmup,
            iterations=iterations,
        )
        forward_rows.append(forward_row)
        dynamics_rows.append(dynamics_row)
        del case
        _release_cuda()
    return forward_rows, dynamics_rows


def _print_metadata(
    arguments: argparse.Namespace,
    package: Any,
    device: torch.device,
    batches: tuple[int, ...],
) -> None:
    """Print the environment and exact source/fixture identities."""
    better_robot_root = Path(br.__file__).resolve().parents[2]
    better_robot_revision, better_robot_dirty = _git_state(better_robot_root)
    better_human_revision, better_human_dirty = _git_state(ROOT)
    print(f"torch={torch.__version__}")
    print(f"better_robot={br.__version__}")
    print(f"better_robot_revision={better_robot_revision}; dirty={better_robot_dirty}")
    print(f"better_robot_source={better_robot_root}")
    print(f"better_human_revision={better_human_revision}; dirty={better_human_dirty}")
    print(f"smplx_reference=0.1.28 ({Path(package.__file__).resolve().parent})")
    print(f"warp={version('warp-lang')}")
    print(f"device=cuda ({torch.cuda.get_device_name(device)})")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"cuda_driver={_cuda_driver_version()}")
    print(f"warmup={arguments.warmup}; iterations={arguments.iterations}; batches={batches}")
    print("dtype=torch.float32; correctives=MHR on; compile=False")
    print("timing_caveat=shared-host scheduling and concurrent load may affect medians")
    print(f"smpl_path={arguments.smpl_path.resolve()}")
    print(f"smplx_path={arguments.smplx_path.resolve()}")
    print(f"mhr_path={arguments.mhr_path.resolve()}")
    print(f"mhr_reference_path={arguments.mhr_reference_path.resolve()}")


def main() -> None:
    """Run the compact all-body benchmark and print two Markdown tables."""
    arguments = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("this compact benchmark requires CUDA")
    device = torch.device("cuda")
    for name, path in (
        ("SMPL", arguments.smpl_path),
        ("SMPL-X", arguments.smplx_path),
        ("MHR", arguments.mhr_path),
        ("MHR reference", arguments.mhr_reference_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} fixture is absent: {path}")

    package, struct_type = _vendored_smplx()
    batches = tuple(arguments.batches)
    _print_metadata(arguments, package, device, batches)

    forward_rows: list[ForwardRow] = []
    dynamics_rows: list[DynamicsRow] = []
    with torch.inference_mode():
        smpl = bh.SMPL(arguments.smpl_path, num_betas=10, compute_mass=True, device=device)
        family_forward, family_dynamics = _measure_batches(
            lambda batch: _prepare_smpl_case(
                "SMPL",
                arguments.smpl_path,
                smpl,
                package,
                struct_type,
                batch,
                device,
            ),
            batches,
            warmup=arguments.warmup,
            iterations=arguments.iterations,
        )
        forward_rows.extend(family_forward)
        dynamics_rows.extend(family_dynamics)
        smpl = None
        _release_cuda()

        smplx = bh.SMPLX(
            arguments.smplx_path,
            num_betas=16,
            num_expression=10,
            use_hands=True,
            use_face=True,
            flat_hand_mean=False,
            compute_mass=True,
            device=device,
        )
        family_forward, family_dynamics = _measure_batches(
            lambda batch: _prepare_smpl_case(
                "SMPL-X",
                arguments.smplx_path,
                smplx,
                package,
                struct_type,
                batch,
                device,
            ),
            batches,
            warmup=arguments.warmup,
            iterations=arguments.iterations,
        )
        forward_rows.extend(family_forward)
        dynamics_rows.extend(family_dynamics)
        smplx = None
        _release_cuda()

        mhr_reference = torch.jit.load(arguments.mhr_reference_path, map_location=device)
        mhr_reference.eval()
        mhr = bh.MHR(
            arguments.mhr_path,
            lod=1,
            use_correctives=True,
            compute_mass=True,
            device=device,
        )
        family_forward, family_dynamics = _measure_batches(
            lambda batch: _prepare_mhr_case(mhr, mhr_reference, batch, device),
            batches,
            warmup=arguments.warmup,
            iterations=arguments.iterations,
        )
        forward_rows.extend(family_forward)
        dynamics_rows.extend(family_dynamics)
        mhr = mhr_reference = None
        _release_cuda()

    _print_results(forward_rows, dynamics_rows)


if __name__ == "__main__":
    main()
