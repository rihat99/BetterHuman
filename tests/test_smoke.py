from __future__ import annotations

from pathlib import Path

import torch

import better_human
import better_robot
from better_robot import Model
from better_robot.io.builders.kinematic_tree import build_kinematic_tree_model


def _free_flyer_spherical_tree() -> Model:
    """Weight-free 24-body free-flyer + 23-spherical tree (nq=99, nv=75)."""
    joint_names = tuple(f"j{index}" for index in range(24))
    parents = (-1, *range(23))
    translations = torch.zeros(24, 3)
    return build_kinematic_tree_model(
        name="free_flyer_spherical_tree",
        joint_names=joint_names,
        parents=parents,
        translations=translations,
        root_kind="free_flyer",
        child_kind="spherical",
        preserve_joint_order=True,
    )


def test_foundation_imports_from_editable_checkouts() -> None:
    assert better_human.__version__ == "0.1.0.dev0"
    assert better_robot.__file__ is not None

    sibling_source = Path(__file__).resolve().parents[2] / "BetterRobot" / "src"
    assert Path(better_robot.__file__).resolve().is_relative_to(sibling_source.resolve())


def test_free_flyer_spherical_forward_kinematics_is_batched() -> None:
    model = _free_flyer_spherical_tree()
    data = better_robot.forward_kinematics(model, model.q_neutral)

    assert model.nq == 99
    assert model.nv == 75
    assert data.joint_pose_world.shape == (25, 7)

    q_batch = model.q_neutral.expand(4, -1).clone()
    batch_data = better_robot.forward_kinematics(model, q_batch)
    assert batch_data.joint_pose_world.shape == (4, 25, 7)
