"""The benchmark runner: counters and wall time of BetterHuman's hot entry points, with the context to read them.

Usage (from the repository root)::

    uv run python benchmarks/run.py --counters-only     # the counter rows, checked against the thresholds
    uv run python benchmarks/run.py                     # the counter rows with wall time
    NUMPY_MADVISE_HUGEPAGE=0 uv run python benchmarks/run.py --licensed   # also the sweep and the load rows

Counter rows (principle 5.8) run on the tracked random-weight archives (``tests/support/random_weights``), so the gate
needs no licensed weights. Those archives have the families' vertex and joint counts but a random kinematic tree and
face strip: graph breaks, recompiles and host syncs equal the licensed archives', the operator counts do not. Rows:
``fk_vertices/smplx/B{1,64}`` (``identity.forward(q).vertices`` of one baked SMPL-X identity without part inertias),
``fk_streams/smplx/B{1,64}`` (the same with the expression stream), ``joint_poses/smplx/B{1,64}``
(``identity.forward(q).joint_poses``), ``contact_frame_poses/smplx/B{1,64}``
(``identity.forward(q).contact_frame_poses(CONTACT_FRAMES)``, the identity's placements cached by the warm-up call),
``to_classic/smplx/B{1,64}`` (``better_human.to_classic(identity, q)``), ``bake_forward/smpl/B{1,64}``
(``body.bake(betas=betas).forward(q).vertices``, one identity per pose, ``density=None``) and
``bake_forward_dempster/smpl/B{1,64}`` (the same with the default ``density="Dempster"``: 24 part inertias per bake), in
the neutral pose with seeded identity coefficients. Each has four counters, deterministic for one torch version and
device: ``graph_breaks`` (``torch._dynamo.explain``), ``recompiles`` (frames compiled when ``torch.compile(fn,
backend="eager")``, first called at the row's batch, is called at each batch of ``SWEEP``), ``ops`` (top-level
``aten::`` operators of one call) and ``host_syncs`` (calls of the operators that make the host wait for a CUDA queue),
the last two from ``better_robot.testing.bench.profile_counts``, which BetterRobot's runner shares; the first two need the
compiler, which BetterRobot's contract C-PERF-5 keeps out of its library, so ``compile_counts`` stays here. The run
exits 1 when a counter row exceeds its threshold in
``benchmarks/thresholds/torch-<version>.json``, has none, or no such file exists for the installed torch; a thresholds
file is this runner's ``--counters-only`` output from a clean commit, changed only in a commit of its own.

``--licensed`` adds the rows that read licensed weights through ``better_human.io`` (``$BETTERHUMAN_MODELS_DIR``). Sweep
rows (principle 5.7) of SMPL, SMPL-X, MHR (LOD 1) and SOMA, ``<family>/<op>/B<batch>/<variant>`` (the earlier runner's
keys, so its result files compare row by row): ``bake`` at batch 1 with part inertias (``mass``, the default density)
and without (``no_mass``); at each ``--batches`` size, ``vertices`` under ``no_grad`` and ``vertices_backward`` (the
gradient of ``sum(vertices)`` with respect to ``q``) of one baked identity (``default``: no stream, except MHR's
expression; SMPL-X also with its expression stream moving the joints, ``expression_stream_joints``, and not,
``expression_frozen_joints``; the default identity's ``vertices`` also through ``better_robot.kinematics.compiled``,
``compiled``, timed after the two warm-up calls in which it runs eagerly and compiles, without its backward: torch
2.13's compiled MHR backward aborts the process on CPU), and BetterRobot's ``fk`` and ``rnea`` (on a precomputed state)
on the baked model; MHR's parity oracle, its released TorchScript model, beside them (variant ``torchscript``). Load
rows ``load/{smpl,smplx,mhr_lod1,soma}``: the family loader's time beside the raw read of the same archive (each
member's bytes, then ``np.load`` of every member without object dtype) and their ``ratio``.

Wall time is the median of up to ``--repeats`` calls (at least 3; fewer once a row has run 2 s) after one
warm-up call, with the minor page faults per call; the counters are taken after it. It is recorded, never gated: the
load average sits beside every row. For rows at batch >= ``LARGE_BATCH`` the runner re-executes itself with glibc's
allocator thresholds fixed (``FIXED_MALLOC``; they must be set before the process starts) and records that time and
its page faults beside the default ones: the default allocator's dynamic mmap threshold depends on what the process
freed before. On a host with transparent huge pages a load can stall for seconds (``CLAUDE.md``, "What the tiers
need"), so time loads with ``NUMPY_MADVISE_HUGEPAGE=0``; ``meta`` records the setting.

Every number is float32, ``--threads`` Torch threads (default 1), ``BETTER_ROBOT_INTERNAL_CHECKS`` off. Output:
``{"meta": {...}, "rows": [{"row": id, ...}, ...]}`` in ``--out`` (default
``benchmarks/results/<commit>_<device>.json``, never tracked); ``meta`` records commit, dirty flag, command, the
BetterRobot file imported, torch, Python, device, CPU, threads, affinity, the allocator and huge-page variables and the
load average at start and end.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import better_robot as br
import numpy as np
import torch
import torch._dynamo
from better_robot.testing import bench
from torch._dynamo.utils import counters as dynamo_counters

import better_human as bh

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVES = REPO_ROOT / "tests" / "support" / "random_weights"  # tracked random weights: no licence
THRESHOLDS = REPO_ROOT / "benchmarks" / "thresholds" / f"torch-{torch.__version__.split('+')[0]}.json"
DTYPE = torch.float32
SWEEP = (1, 2, 3, 64)
#: Two frames whose parents the frame check resolves from the part ids (the ``contact_frame_poses`` rows).
CONTACT_FRAMES = {"left_heel": (8846, None), "right_heel": (8635, None)}
LARGE_BATCH = 1024
FIXED_MALLOC = {"MALLOC_MMAP_THRESHOLD_": "134217728", "MALLOC_TRIM_THRESHOLD_": "268435456"}
#: load row -> the archive under ``$BETTERHUMAN_MODELS_DIR`` (folder, file) and the family loader that reads it.
LOADS: dict[str, tuple[tuple[str, str], Callable[..., bh.FamilySpec]]] = {
    "smpl": (("smpl", "SMPL_NEUTRAL.npz"), bh.families.smpl),
    "smplx": (("smplx", "SMPLX_NEUTRAL.npz"), bh.families.smplx),
    "mhr_lod1": (("MHR/converted", "mhr_lod1.npz"), partial(bh.families.mhr, lod=1)),
    "soma": (("SOMA", "soma_mid.npz"), bh.families.soma),
}


@dataclass(frozen=True)
class Case:
    """One row: ``call()`` is timed; a counter row also profiles it and compiles ``fn`` at ``args`` and ``sweep``."""

    row: str
    batch: int
    call: Callable[[], Any]
    fields: dict[str, Any]
    fn: Callable[..., Any] | None = None
    args: tuple[torch.Tensor, ...] = ()
    sweep: tuple[tuple[torch.Tensor, ...], ...] = ()
    warmups: int = 1


def identity_coefficients(spec: bh.FamilySpec, device: torch.device) -> dict[str, torch.Tensor]:
    """Seeded ``(n,)`` coefficients of every identity group of ``spec``."""
    generator = torch.Generator().manual_seed(1)
    groups = spec.identity_groups.items()
    return {name: (0.5 * torch.randn(width, generator=generator)).to(device) for name, width in groups}


def counter_cases(device: torch.device) -> list[Case]:
    """The counter rows (module docstring), with seeded inputs on ``device``."""
    smplx = bh.HumanModel.from_spec(bh.families.smplx(ARCHIVES / "smplx_random.npz", device=device), density=None)
    identity = smplx.bake(**identity_coefficients(smplx.spec, device))
    pool = identity.model.q_neutral.expand(64, identity.model.nq).clone()

    generator = torch.Generator().manual_seed(1)
    width = smplx.spec.stream_groups["expression"]
    expression_pool = (0.5 * torch.randn(64, width, generator=generator)).to(device)

    def fk_vertices(q: torch.Tensor) -> torch.Tensor:
        return identity.forward(q).vertices

    def fk_streams(q: torch.Tensor, expression: torch.Tensor) -> torch.Tensor:
        return identity.forward(q, {"expression": expression}).vertices

    def joint_poses(q: torch.Tensor) -> torch.Tensor:
        return identity.forward(q).joint_poses

    def contact_frame_poses(q: torch.Tensor) -> torch.Tensor:
        return identity.forward(q).contact_frame_poses(CONTACT_FRAMES)

    def to_classic(q: torch.Tensor) -> Any:
        return bh.to_classic(identity, q)

    rows = {
        "fk_vertices": (fk_vertices, (pool,)),
        "fk_streams": (fk_streams, (pool, expression_pool)),
        "joint_poses": (joint_poses, (pool,)),
        "contact_frame_poses": (contact_frame_poses, (pool,)),
        "to_classic": (to_classic, (pool,)),
    }
    cases = [
        counter_case(f"{name}/smplx/B{batch}", fn, pools, batch)
        for name, (fn, pools) in rows.items()
        for batch in (1, 64)
    ]
    smpl = bh.families.smpl(ARCHIVES / "smpl_random.npz", device=device)
    for prefix, density in (("bake_forward", None), ("bake_forward_dempster", "Dempster")):
        body = bh.HumanModel.from_spec(smpl, density=density)
        generator = torch.Generator().manual_seed(1)
        betas = (0.5 * torch.randn(64, smpl.identity_groups["betas"], generator=generator)).to(device)
        q_pool = body.model.q_neutral.expand(64, body.model.nq).clone()

        def bake_forward(betas: torch.Tensor, q: torch.Tensor, body: bh.HumanModel = body) -> torch.Tensor:
            return body.bake(betas=betas).forward(q).vertices

        cases += [counter_case(f"{prefix}/smpl/B{batch}", bake_forward, (betas, q_pool), batch) for batch in (1, 64)]
    return cases


def counter_case(row: str, fn: Callable[..., Any], pools: tuple[torch.Tensor, ...], batch: int) -> Case:
    """A counter row of ``fn`` on the first ``batch`` entries of each pool, swept over ``SWEEP``."""
    args = tuple(pool[:batch] for pool in pools)
    sweep = tuple(tuple(pool[:size] for pool in pools) for size in SWEEP)
    return Case(row, batch, lambda: fn(*args), {}, fn, args, sweep)


def family_spec(family: str, device: torch.device) -> bh.FamilySpec:
    """The sweep's ``family`` from the licensed weights, with the earlier runner's options."""
    if family == "SMPL":
        return bh.families.smpl(num_betas=10, device=device)
    if family == "SMPL-X":
        return bh.families.smplx(num_betas=10, num_expression=10, device=device)
    if family == "MHR":
        return bh.families.mhr(lod=1, device=device)
    return bh.families.soma(device=device)


def sweep_cases(args: argparse.Namespace, device: torch.device) -> list[Case]:
    """The sweep rows (module docstring) of every family, seeded inputs on ``device``."""
    cases = []
    for family in ("SMPL", "SMPL-X", "MHR", "SOMA"):
        spec = family_spec(family, device)
        coefficients = identity_coefficients(spec, device)
        bodies = {"mass": bh.HumanModel.from_spec(spec), "no_mass": bh.HumanModel.from_spec(spec, density=None)}
        for variant, body in bodies.items():
            cases.append(sweep_case(family, "bake", 1, variant, partial(body.bake, **coefficients)))
        identity = bodies["mass"].bake(**coefficients)
        identities = {"default": (identity, family == "MHR")}
        if family == "SMPL-X":
            frozen = bh.HumanModel.from_spec(spec, density=None, stream_joints=False).bake(**coefficients)
            identities |= {"expression_stream_joints": (identity, True), "expression_frozen_joints": (frozen, True)}
        forwards = {
            variant: (partial(posed_vertices, posed), with_streams)
            for variant, (posed, with_streams) in identities.items()
        }
        forwards["compiled"] = (br.kinematics.compiled(forwards["default"][0]), forwards["default"][1])
        model = identity.model
        # torch 2.13 deprecates torch.jit.load and still loads the released file
        oracle = (
            torch.jit.load(bh.io.weight_path("MHR/assets", "mhr_model.pt"), map_location=device)
            if family == "MHR"
            else None
        )
        for batch in args.batches:
            q = model.q_neutral.expand(batch, model.nq).clone()
            generator = torch.Generator().manual_seed(2)
            streams = {
                name: (0.3 * torch.randn(batch, width, generator=generator)).to(device)
                for name, width in spec.stream_groups.items()
            }
            for variant, (forward, with_streams) in forwards.items():
                for op in ("vertices",) if variant == "compiled" else ("vertices", "vertices_backward"):
                    call = partial(vertices, forward, q, streams if with_streams else None, backward=op != "vertices")
                    cases.append(
                        sweep_case(family, op, batch, variant, call, warmups=2 if variant == "compiled" else 1)
                    )
            generator = torch.Generator().manual_seed(3)
            v, a = ((0.1 * torch.randn(batch, model.nv, generator=generator)).to(device) for _ in range(2))
            state = br.forward_kinematics(model, q)
            cases.append(sweep_case(family, "fk", batch, "default", partial(no_grad, br.forward_kinematics, model, q)))
            rnea = partial(no_grad, br.inverse_dynamics, model, state, v=v, a=a)
            cases.append(sweep_case(family, "rnea", batch, "default", rnea))
            if oracle is not None:
                cases.append(sweep_case(family, "vertices", batch, "torchscript", mhr_oracle(oracle, batch, device)))
    return cases


def sweep_case(family: str, op: str, batch: int, variant: str, call: Callable[[], Any], *, warmups: int = 1) -> Case:
    """A sweep row ``<family>/<op>/B<batch>/<variant>``."""
    fields = {"family": family, "op": op, "variant": variant}
    return Case(f"{family}/{op}/B{batch}/{variant}", batch, call, fields, warmups=warmups)


def posed_vertices(
    identity: bh.Identity, q: torch.Tensor, streams: dict[str, torch.Tensor] | None = None
) -> torch.Tensor:
    """``identity.forward(q, streams).vertices``: the function a ``compiled`` row compiles."""
    return identity.forward(q, streams).vertices


def vertices(
    forward: Callable[..., torch.Tensor], q: torch.Tensor, streams: dict[str, torch.Tensor] | None, *, backward: bool
) -> Any:
    """``forward(q, streams)`` (``(B, V, 3)`` vertices) under ``no_grad``, or the gradient of their sum with respect to
    ``q (B, nq)``."""
    if backward:
        leaf = q.detach().requires_grad_(True)
        return torch.autograd.grad(forward(leaf, streams).sum(), leaf)
    return no_grad(forward, q, streams)


def no_grad(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``fn(*args, **kwargs)`` under ``torch.no_grad``."""
    with torch.no_grad():
        return fn(*args, **kwargs)


def mhr_oracle(model: torch.jit.ScriptModule, batch: int, device: torch.device) -> Callable[[], torch.Tensor]:
    """``(B, V, 3)`` vertices of MHR's released TorchScript ``model`` at fixed identity, pose and expression inputs."""
    inputs = (
        torch.linspace(-0.02, 0.02, 45, device=device).repeat(batch, 1),
        torch.linspace(-0.04, 0.04, 204, device=device).repeat(batch, 1),
        torch.linspace(0.015, -0.015, 72, device=device).repeat(batch, 1),
    )
    return partial(no_grad, lambda: model(*inputs, True)[0])


def raw_read(path: Path) -> dict[str, np.ndarray]:
    """Every member's bytes, then ``np.load`` of each member whose dtype holds no Python objects (those need pickle)."""
    headers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            buffer = io.BytesIO(archive.read(name))
            _, _, dtype = headers[np.lib.format.read_magic(buffer)](buffer)
            if not dtype.hasobject:
                buffer.seek(0)
                arrays[name] = np.load(buffer, allow_pickle=False)
    return arrays


def load_rows(args: argparse.Namespace, device: torch.device, sync: Callable[[], None]) -> list[dict[str, Any]]:
    """The load rows: the loader and the raw read of one archive, each timed after one warm-up call."""
    rows = []
    for name, ((folder, file), loader) in LOADS.items():
        row: dict[str, Any] = {"row": f"load/{name}", "loadavg": bench.load_average()}
        if args.rows is not None and row["row"] not in args.rows:
            continue
        path = bh.io.weight_path(folder, file)
        load, read = partial(loader, path, device=device), partial(raw_read, path)
        load()  # warm-ups
        read()
        row |= bench.wall_time(load, sync, repeats=args.repeats)
        row["raw_read_ms"] = bench.wall_time(read, lambda: None, repeats=args.repeats)["median_ms"]
        row["ratio"] = row["median_ms"] / row["raw_read_ms"]
        rows.append(row)
        times = f"{row['median_ms']:10.1f} ms, raw read {row['raw_read_ms']:.1f} ms"
        print(f"{row['row']:48} {times}, ratio {row['ratio']:.2f}", flush=True)
    return rows


def compile_counts(fn: Callable[..., Any], args: tuple, sweep: tuple[tuple, ...]) -> tuple[int, int]:
    """``(graph_breaks, recompiles)`` of ``fn`` (module docstring); dynamo state is reset before and after."""
    torch._dynamo.reset()
    graph_breaks = torch._dynamo.explain(fn)(*args).graph_break_count
    torch._dynamo.reset()
    dynamo_counters.clear()
    compiled = torch.compile(fn, backend="eager")
    compiled(*args)
    after_first = dynamo_counters["frames"]["ok"]
    for sweep_args in sweep:
        compiled(*sweep_args)
    recompiles = dynamo_counters["frames"]["ok"] - after_first
    torch._dynamo.reset()
    return graph_breaks, recompiles


def measure(case: Case, args: argparse.Namespace, sync: Callable[[], None]) -> dict[str, Any]:
    """One case's row: its warm-up calls, wall time unless ``--counters-only``, counters unless ``--timing-only``."""
    row: dict[str, Any] = {"row": case.row, "batch": case.batch, **case.fields, "loadavg": bench.load_average()}
    for _ in range(case.warmups):
        case.call()
    if not args.counters_only:
        row |= bench.wall_time(case.call, sync, repeats=args.repeats)
    if not args.timing_only and case.fn is not None:
        row["ops"], row["host_syncs"] = bench.profile_counts(case.call)
        row["graph_breaks"], row["recompiles"] = compile_counts(case.fn, case.args, case.sweep)
    print(bench.row_line(row), flush=True)
    return row


def measure_sweep(cases: list[Case], args: argparse.Namespace, sync: Callable[[], None]) -> list[dict[str, Any]]:
    """Measure the sweep rows; a row that raises records its error, and larger batches of a too-slow row are skipped."""
    rows: list[dict[str, Any]] = []
    slow: dict[tuple, int] = {}  # (family, op, variant) -> batch whose call exceeded --max-call-s
    for case in cases:
        key = (case.fields["family"], case.fields["op"], case.fields["variant"])
        if key in slow:
            error = f"skipped: batch {slow[key]} exceeded --max-call-s"
            rows.append({"row": case.row, "batch": case.batch, **case.fields, "error": error})
            continue
        try:
            rows.append(measure(case, args, sync))
        except Exception as exc:  # noqa: BLE001 - one row's failure (memory at a large batch) is recorded, not fatal
            rows.append({"row": case.row, "batch": case.batch, **case.fields, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{case.row:<48} ERROR {rows[-1]['error'][:80]}")
            continue
        if rows[-1].get("median_ms", 0.0) > args.max_call_s * 1e3:
            slow[key] = case.batch
    return rows


def fixed_malloc_numbers(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Re-run the rows at batch >= ``LARGE_BATCH`` in a child with ``FIXED_MALLOC`` set; add its time and faults."""
    selected = [row["row"] for row in rows if row.get("batch", 0) >= LARGE_BATCH and "median_ms" in row]
    if not selected:
        return
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "fixed.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--timing-only", "--rows", ",".join(selected)]
        command += ["--device", args.device, "--threads", str(args.threads), "--repeats", str(args.repeats)]
        command += ["--batches", ",".join(map(str, args.batches)), "--out", str(out)]
        command += ["--licensed"] if args.licensed else []
        print(f"fixed allocator thresholds ({' '.join(f'{k}={v}' for k, v in FIXED_MALLOC.items())}):", flush=True)
        subprocess.run(command, env={**os.environ, **FIXED_MALLOC}, check=True)
        fixed = {row["row"]: row for row in json.loads(out.read_text())["rows"]}
    for row in rows:
        if row["row"] in fixed and "median_ms" in fixed[row["row"]]:
            row["fixed_malloc_median_ms"] = fixed[row["row"]]["median_ms"]
            row["fixed_malloc_page_faults"] = fixed[row["row"]]["page_faults"]


def parse_args() -> argparse.Namespace:
    """The command line, with the comma lists split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batches", default="1,64,1024", help="sweep batch sizes")
    parser.add_argument("--licensed", action="store_true", help="also the rows that read $BETTERHUMAN_MODELS_DIR")
    parser.add_argument("--repeats", type=int, default=20, help="max timed calls per row (fewer once a row passes 2 s)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rows", default=None, help="measure only these comma-separated row ids")
    parser.add_argument("--max-call-s", type=float, default=20.0, help="skip larger sweep batches past this call time")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--counters-only", action="store_true", help="counter rows only, no wall time; check thresholds")
    mode.add_argument("--timing-only", action="store_true", help="wall time only, no counters")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    args.batches = [int(b) for b in args.batches.split(",")]
    args.rows = None if args.rows is None else set(args.rows.split(","))
    return args


def main() -> None:
    """Measure the rows, write the JSON, check the counters; exit 1 on a counter problem."""
    args = parse_args()
    if os.environ.get("BETTER_ROBOT_INTERNAL_CHECKS") == "1":
        sys.exit("BETTER_ROBOT_INTERNAL_CHECKS=1 adds host syncs and operators: unset it to benchmark")
    torch.set_num_threads(args.threads)  # noqa: TID251 - a benchmark pins its threads so rows compare; C-DEP-3 binds the library
    device = torch.device(args.device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    meta = bench.context(REPO_ROOT, device, dtype=DTYPE)
    # under the repository's name, as BetterHuman cites BetterRobot paths (no lane or home directory in a tracked
    # file): src/better_robot/... from a checkout on PYTHONPATH, site-packages/better_robot/... from the pinned copy
    meta["better_robot_file"] = "BetterRobot/" + os.path.relpath(br.__file__, Path(br.__file__).parents[2])
    meta["env"] = {name: os.environ.get(name) for name in (*FIXED_MALLOC, "NUMPY_MADVISE_HUGEPAGE")}
    started = time.perf_counter()
    print(bench.HEADER)

    cases = [case for case in counter_cases(device) if args.rows is None or case.row in args.rows]
    counter_rows = [measure(case, args, sync) for case in cases]
    rows = list(counter_rows)
    if args.licensed and not args.counters_only:
        cases = [case for case in sweep_cases(args, device) if args.rows is None or case.row in args.rows]
        rows += measure_sweep(cases, args, sync)
        rows += load_rows(args, device, sync)
    if not args.counters_only:
        if not any(os.environ.get(name) for name in FIXED_MALLOC):
            fixed_malloc_numbers(rows, args)

    meta["loadavg_end"] = bench.load_average()
    meta["total_s"] = round(time.perf_counter() - started, 1)
    out = args.out or REPO_ROOT / "benchmarks" / "results" / f"{meta['commit']}_{args.device}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n")
    print(f"wrote {out} ({meta['total_s']} s)")
    if args.timing_only:
        return
    problems = bench.check_thresholds(counter_rows, THRESHOLDS, selected=args.rows)
    print(f"counters against {THRESHOLDS.relative_to(REPO_ROOT)}: {len(problems)} problems")
    for problem in problems:
        print(f"  {problem}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
