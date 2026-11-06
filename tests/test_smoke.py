from __future__ import annotations

import tomllib
from pathlib import Path

import torch

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


def test_better_robot_imports_and_its_declared_checkout_exists() -> None:
    """``better_robot`` imports (this module imports it), and the checkout ``pyproject.toml`` declares for it exists.

    Which copy of ``better_robot`` runs is the environment's choice (a ``PYTHONPATH`` override runs another checkout
    on purpose), so the test does not pin it.
    """
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    source = tomllib.loads(pyproject.read_text())["tool"]["uv"]["sources"]["better-robot"]
    checkout = (pyproject.parent / source["path"]).resolve()
    assert (checkout / "src" / "better_robot" / "__init__.py").is_file(), (
        f"pyproject.toml declares better-robot at {source['path']!r}, but {checkout} holds no BetterRobot checkout; "
        "place BetterRobot there or change [tool.uv.sources]"
    )


def test_free_flyer_spherical_forward_kinematics_is_batched() -> None:
    model = _free_flyer_spherical_tree()
    state = better_robot.forward_kinematics_state(model, model.q_neutral)

    assert model.nq == 99
    assert model.nv == 75
    assert state.joint_pose.shape == (25, 7)

    q_batch = model.q_neutral.expand(4, -1).clone()
    batch_state = better_robot.forward_kinematics_state(model, q_batch)
    assert batch_state.joint_pose.shape == (4, 25, 7)
