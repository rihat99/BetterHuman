from pathlib import Path

import better_human
import better_robot
from better_robot.io.builders.smpl_like import make_smpl_like_model


def test_foundation_imports_from_editable_checkouts() -> None:
    assert better_human.__version__ == "0.1.0.dev0"
    assert better_robot.__file__ is not None

    sibling_source = Path(__file__).resolve().parents[2] / "BetterRobot" / "src"
    assert Path(better_robot.__file__).resolve().is_relative_to(sibling_source.resolve())


def test_smpl_like_forward_kinematics_is_batched() -> None:
    model = make_smpl_like_model(preserve_joint_order=True)
    data = better_robot.forward_kinematics(model, model.q_neutral)

    assert model.nq == 99
    assert model.nv == 75
    assert data.joint_pose_world.shape == (25, 7)

    q_batch = model.q_neutral.expand(4, -1).clone()
    batch_data = better_robot.forward_kinematics(model, q_batch)
    assert batch_data.joint_pose_world.shape == (4, 25, 7)
