"""Dropping terminal joint subtrees (``hands=False``, ``face=False``, ``families.drop_joints``) and the loaders'
blend-group widths, through the public loaders on the random-weight archives and synthetic FLAME; no weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from better_robot.exceptions import ArgumentError

import better_human as bh
from better_human.families import SMPLHClassic, SMPLXClassic
from tests.support.random_archives import random_weight_spec
from tests.support.synthetic import flame_spec, write_family

F64 = torch.float64


@pytest.mark.parametrize(
    ("family", "dropped", "classic", "num_joints"),
    [
        ("smplh", {"hands": False}, SMPLHClassic, 22),
        ("smplx", {"hands": False, "face": False}, SMPLXClassic, 22),
        ("smplx", {"face": False}, SMPLXClassic, 52),  # jaw and eyes sit between the body and the hands
    ],
    ids=["smplh-no-hands", "smplx-no-hands-no-face", "smplx-no-face"],
)
def test_a_body_without_its_resting_hands_and_face_poses_as_the_full_body(
    family: str, dropped: dict[str, bool], classic: type, num_joints: int
) -> None:
    """With the dropped joints at rest, the reduced body equals the full body at any body pose and shape.

    The dropped skin weights go to the surviving ancestor, whose skin affine equals theirs at rest, and their
    corrective features are zero there. Oracle: the full model, float64. Measured 2026-11-28: vertices 4.4e-16 m,
    joints 0 (summation order); weights sent to the wrong joint or left out, or the corrective rows of other joints,
    move vertices by 0.15 m to 1.4 m on these archives; the regressor rows of other joints after a drop inside the
    tree (smplx-no-face) move the hand joints by 0.63 m.
    """
    generator = torch.Generator().manual_seed(0)
    fields = {
        name: 0.3 * torch.randn(2, width, generator=generator, dtype=F64)
        for name, width in (("betas", 16), ("global_orient", 3), ("body_pose", 63), ("transl", 3))
    }
    options = {"flat_hand_mean": True} if family == "smplx" else {}  # a zero classic hand pose is then at rest
    states, specs = [], []
    for spec in (
        random_weight_spec(family, dtype=F64, **options),
        random_weight_spec(family, dtype=F64, **options, **dropped),
    ):
        identity, q, streams = bh.from_classic(bh.HumanModel.from_spec(spec, density=None), classic(**fields))
        states.append(identity.forward(q, streams))
        specs.append(spec)
    (full, reduced), (full_spec, reduced_spec) = states, specs
    assert reduced_spec.num_joints == num_joints
    kept = [full_spec.joint_id(name) for name in reduced_spec.joint_names]
    torch.testing.assert_close(reduced.vertices, full.vertices, rtol=0, atol=1e-14)
    torch.testing.assert_close(reduced.joint_translations, full.joint_translations[..., kept, :], rtol=0, atol=1e-14)


def test_drop_joints_refuses_the_root_a_subtree_left_open_and_a_vertex_without_weights(tmp_path: Path) -> None:
    spec = flame_spec(tmp_path)  # global, neck, jaw, left_eye, right_eye; neck carries the other three
    with pytest.raises(ArgumentError, match="root joint cannot be dropped"):
        bh.families.drop_joints(spec, ["global"])
    with pytest.raises(ArgumentError, match="joint 1 is dropped but child 2 survives"):
        bh.families.drop_joints(spec, ["neck"])
    with pytest.raises(ArgumentError, match=r"out of range \[0, 5\): \[5\]"):
        spec.rig.without_joints(spec, [5])
    path = write_family(tmp_path, "flame")
    arrays = dict(np.load(path))
    arrays["weights"][0] = 0.0
    np.savez(path, **arrays)
    with pytest.raises(ArgumentError, match="positive finite sums"):
        bh.families.drop_joints(bh.families.flame(path, dtype=F64), ["jaw"])


def test_a_dropped_joint_weight_goes_to_its_surviving_parent_and_every_row_is_renormalised(tmp_path: Path) -> None:
    """Dropping FLAME's jaw from a file whose weight rows sum to 0.9 (a loader takes rows as stored).

    Worked by hand for vertex 0, weights (global, neck, jaw, left_eye, right_eye) = (0.1, 0.2, 0.3, 0.2, 0.1): the
    jaw's 0.3 joins the neck's 0.2, and the row (0.1, 0.5, 0.2, 0.1) is divided by its sum 0.9. Float64, 1e-15.
    """
    path = write_family(tmp_path, "flame")
    arrays = dict(np.load(path))
    arrays["weights"] = 0.9 * arrays["weights"]
    arrays["weights"][0] = [0.1, 0.2, 0.3, 0.2, 0.1]
    np.savez(path, **arrays)
    reduced = bh.families.drop_joints(bh.families.flame(path, dtype=F64), ["jaw"])
    assert reduced.joint_names == ("global", "neck", "left_eye", "right_eye")
    dense = torch.zeros(len(reduced.skin_weights), 4, dtype=F64)
    dense = dense.scatter_add(1, reduced.skin_indices, reduced.skin_weights)  # (V, 4)
    torch.testing.assert_close(dense.sum(dim=-1), torch.ones(len(dense), dtype=F64), rtol=0, atol=1e-15)
    torch.testing.assert_close(dense[0], torch.tensor([1.0, 5.0, 2.0, 1.0], dtype=F64) / 9, rtol=0, atol=1e-15)


def test_a_blend_group_wider_than_the_file_raises() -> None:
    with pytest.raises(ArgumentError, match="num_betas=11 exceeds the 10 betas columns"):
        random_weight_spec("smpl", num_betas=11)
