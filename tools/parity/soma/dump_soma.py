"""Dump deterministic float32 SOMA-X parity goldens (mid LOD, metres) from the reference package.

Run with the upstream reference environment and ``CUDA_VISIBLE_DEVICES=""`` (warp enumerates GPUs otherwise).
Writes one ``np.savez`` archive per case: float32 unless noted, metres, y-up, column-vector 4x4 transforms
(``M @ [p; 1]``), joints in the converted archive's order (V = 18056 vertices, J = 110 skin joints, P = 78 public):

- ``format_version`` () int64 = 2, ``case_name`` () str, ``seed`` () int64, ``apply_correctives`` () bool.
- ``identity_coeffs`` (128,); ``poses`` (77, 3) axis-angle of the public joints after Root, relative to the T-pose;
  ``transl`` (3,) world position of Hips.
- ``vertices`` (V, 3), ``joints`` (77, 3), ``transforms`` (P, 4, 4): the outputs of ``SOMALayer.pose``.
- ``skin_transforms`` (J, 4, 4): posed world transforms of all skin joints, twist helpers included, as passed to
  linear blend skinning (new in version 2).
- ``rest_shape`` (V, 3), ``bind_transforms`` (J, 4, 4): reposed rest mesh and bind transforms cached by
  ``SOMALayer.prepare_identity``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from soma import SOMALayer
from soma.units import Unit

DEFAULT_ASSETS_DIR = Path(
    "/tmp/claude-1003/-home-rikhat-akizhanov-better/364c020a-cd00-4ea3-ab3a-688eb158660a/scratchpad/soma_assets"
)
FORMAT_VERSION = 2
BASE_SEED = 20261030
SHAPE_COUNT = 128
POSE_JOINT_COUNT = 77
VERTEX_COUNT = 18056
PUBLIC_JOINT_COUNT = 78
JOINT_COUNT = 110
POSE_SIGMA = 0.4
TRANSL_SIGMA = 0.5
BIG_SHAPE_SIGMA = 2.5
TWIST_ANGLE_RANGE = (1.0, 1.5)
# Pose rotations are relative to the T-pose world axes: arm bones lie along x, leg bones along -y.
ARM_TWISTS = (("LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand"), 0)
LEG_TWISTS = (("LeftLeg", "LeftShin", "LeftFoot", "RightLeg", "RightShin", "RightFoot"), 1)

LOGGER = logging.getLogger(__name__)


class _Case(NamedTuple):
    identity_coeffs: np.ndarray
    poses: np.ndarray
    transl: np.ndarray
    apply_correctives: bool
    seed: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR, help="Directory with SOMA-X assets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/parity/goldens/soma_mid"),
        help="Directory for the golden archives",
    )
    return parser.parse_args()


def _random_case(
    seed: int, *, identity_sigma: float, pose_sigma: float, transl_sigma: float, apply_correctives: bool = True
) -> _Case:
    generator = np.random.default_rng(seed)
    identity = identity_sigma * generator.standard_normal(SHAPE_COUNT)
    poses = pose_sigma * generator.standard_normal((POSE_JOINT_COUNT, 3))
    transl = transl_sigma * generator.standard_normal(3)
    return _Case(
        identity.astype(np.float32), poses.astype(np.float32), transl.astype(np.float32), apply_correctives, seed
    )


def _twist_case(seed: int, public_joint_names: tuple[str, ...], joints: tuple[str, ...], axis: int) -> _Case:
    """Random identity, zero pose except large single-axis twists about the named bones."""
    generator = np.random.default_rng(seed)
    identity = generator.standard_normal(SHAPE_COUNT).astype(np.float32)
    poses = np.zeros((POSE_JOINT_COUNT, 3), dtype=np.float32)
    for name in joints:
        angle = generator.uniform(*TWIST_ANGLE_RANGE) * generator.choice((-1.0, 1.0))
        poses[public_joint_names.index(name) - 1, axis] = angle
    return _Case(identity, poses, np.zeros(3, dtype=np.float32), True, seed)


def _big_shape_case(seed: int) -> _Case:
    """Every shape coefficient at +-2.5 sigma, plus a random pose."""
    generator = np.random.default_rng(seed)
    identity = BIG_SHAPE_SIGMA * generator.choice((-1.0, 1.0), size=SHAPE_COUNT)
    poses = POSE_SIGMA * generator.standard_normal((POSE_JOINT_COUNT, 3))
    return _Case(identity.astype(np.float32), poses.astype(np.float32), np.zeros(3, dtype=np.float32), True, seed)


def _reflect_case(seed: int) -> _Case:
    """A 5-sigma identity whose skeleton fit takes the Kabsch reflection branch (a covariance with det < 0).

    The identity is the review search's pick (most negative covariance determinant among 64 identities): row 40 of
    the third float64 ``torch.randn(64, 128)`` draw from seed 5, times 5. Pose and transl are random.
    """
    search = torch.Generator().manual_seed(5)
    for _ in range(2):
        torch.randn(64, SHAPE_COUNT, generator=search, dtype=torch.float64)
    identity = 5.0 * torch.randn(64, SHAPE_COUNT, generator=search, dtype=torch.float64)[40]
    generator = np.random.default_rng(seed)
    poses = POSE_SIGMA * generator.standard_normal((POSE_JOINT_COUNT, 3))
    transl = TRANSL_SIGMA * generator.standard_normal(3)
    return _Case(identity.numpy().astype(np.float32), poses.astype(np.float32), transl.astype(np.float32), True, seed)


def _cases(public_joint_names: tuple[str, ...]) -> dict[str, _Case]:
    """The 14 golden cases; case k uses seed BASE_SEED + k."""
    names = [
        "zero",
        *(f"identity_{index:02d}" for index in range(3)),
        *(f"pose_{index:02d}" for index in range(5)),
        "twist_00",
        "twist_01",
        "correctives_off",
        "big_shape",
        "reflect_00",
    ]
    seeds = {name: BASE_SEED + index for index, name in enumerate(names)}
    cases = {"zero": _random_case(seeds["zero"], identity_sigma=0.0, pose_sigma=0.0, transl_sigma=0.0)}
    for name in names[1:4]:
        cases[name] = _random_case(seeds[name], identity_sigma=1.0, pose_sigma=0.0, transl_sigma=0.0)
    for name in names[4:9]:
        cases[name] = _random_case(seeds[name], identity_sigma=1.0, pose_sigma=POSE_SIGMA, transl_sigma=TRANSL_SIGMA)
    cases["twist_00"] = _twist_case(seeds["twist_00"], public_joint_names, *ARM_TWISTS)
    cases["twist_01"] = _twist_case(seeds["twist_01"], public_joint_names, *LEG_TWISTS)
    cases["correctives_off"] = _random_case(
        seeds["correctives_off"],
        identity_sigma=1.0,
        pose_sigma=POSE_SIGMA,
        transl_sigma=TRANSL_SIGMA,
        apply_correctives=False,
    )
    cases["big_shape"] = _big_shape_case(seeds["big_shape"])
    cases["reflect_00"] = _reflect_case(seeds["reflect_00"])
    return cases


def _record_skin_transforms(layer: SOMALayer) -> dict[str, torch.Tensor]:
    """Wrap the layer's skinning so the world transforms of its latest call stay readable under "skin_transforms"."""
    record: dict[str, torch.Tensor] = {}
    skin = layer.batched_skinning.linear_blend_skinning

    def recording_skin(world_transforms: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        record["skin_transforms"] = world_transforms
        return skin(world_transforms, **kwargs)

    layer.batched_skinning.linear_blend_skinning = recording_skin
    return record


def _evaluate(
    layer: SOMALayer, case: _Case, record: dict[str, torch.Tensor]
) -> tuple[dict[str, np.ndarray], float, float]:
    """Run upstream prepare_identity + pose; return float32 outputs and both timings in seconds."""
    start = time.perf_counter()
    layer.prepare_identity(torch.from_numpy(case.identity_coeffs)[None])
    prepared = time.perf_counter()
    record.clear()
    output = layer.pose(
        torch.from_numpy(case.poses)[None],
        transl=torch.from_numpy(case.transl)[None],
        apply_correctives=case.apply_correctives,
    )
    posed = time.perf_counter()
    arrays = {
        "vertices": output.vertices[0],
        "joints": output.joints[0],
        "transforms": output.transforms[0],
        "skin_transforms": record["skin_transforms"][0],
        "rest_shape": layer._cached_rest_shape[0],
        "bind_transforms": layer._cached_bind_transforms_world[0],
    }
    expected_shapes = {
        "vertices": (VERTEX_COUNT, 3),
        "joints": (POSE_JOINT_COUNT, 3),
        "transforms": (PUBLIC_JOINT_COUNT, 4, 4),
        "skin_transforms": (JOINT_COUNT, 4, 4),
        "rest_shape": (VERTEX_COUNT, 3),
        "bind_transforms": (JOINT_COUNT, 4, 4),
    }
    for key, value in arrays.items():
        if value.dtype != torch.float32 or tuple(value.shape) != expected_shapes[key]:
            raise ValueError(f"Unexpected upstream {key}: {value.dtype} {tuple(value.shape)}")
    return {key: value.detach().cpu().numpy() for key, value in arrays.items()}, prepared - start, posed - prepared


def main() -> None:
    """Evaluate the fixed SOMA cases with upstream SOMALayer and write one archive per case."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    torch.manual_seed(BASE_SEED)
    layer = SOMALayer(
        data_root=args.assets_dir,
        device="cpu",
        identity_model_type="soma",
        mode="dense",
        output_unit=Unit.METERS,
    )
    record = _record_skin_transforms(layer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = _cases(layer.public_joint_names)
    with torch.no_grad():
        for name, case in cases.items():
            arrays, prepare_seconds, pose_seconds = _evaluate(layer, case, record)
            np.savez(
                args.output_dir / f"{name}.npz",
                format_version=np.asarray(FORMAT_VERSION, dtype=np.int64),
                case_name=np.asarray(name),
                seed=np.asarray(case.seed, dtype=np.int64),
                identity_coeffs=case.identity_coeffs,
                poses=case.poses,
                transl=case.transl,
                apply_correctives=np.asarray(case.apply_correctives),
                **arrays,
            )
            LOGGER.info("%-16s prepare_identity %.3f s, pose %.3f s", name, prepare_seconds, pose_seconds)
    LOGGER.info("Wrote %d SOMA goldens to %s", len(cases), args.output_dir)


if __name__ == "__main__":
    main()
