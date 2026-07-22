"""Bake MHR shape state and evaluate its pose corrective.

The initial body values are placeholders: a shape bake overwrites robot joint
placements and derives scaled native poses before deformation.  Authored FBX
inverse binds remain the authoritative skinning reference at every shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import better_robot as br
import torch
from better_robot import Data, Model
from better_robot.data_model.reduced_coordinates import expand_configuration

from better_human.core import BodyValues, ParametricBody
from better_human.core.mass import body_inertias_from_shape
from better_human.core.pipeline import (
    blend_shapes,
    lbs,
    mlp_corrective,
    pose_feature_6d_from_euler,
    skinning_transforms_from_inverse,
)

from . import tree


def bake_mhr_shape(
    body: ParametricBody,
    expanded_parameters: Mapping[str, torch.Tensor],
) -> tuple[BodyValues, Model]:
    identity = expanded_parameters["identity"]
    proportions = expanded_parameters["proportions"]
    v_shaped = body.values.v_template + blend_shapes(identity, body._identity_basis)
    if body._use_expression:
        v_shaped = v_shaped + blend_shapes(expanded_parameters["expression"], body._expression_basis)

    translation_delta = torch.einsum("...p,jcp->...jc", proportions, body._translation_basis)
    local_log_scales = torch.einsum("...p,jp->...j", proportions, body._scale_basis)
    joint_scales = tree.accumulated_scales(local_log_scales, body.structure.parents)
    parent_indices = torch.tensor(body.structure.parents[1:], dtype=torch.long, device=proportions.device)
    parent_scales = joint_scales.index_select(-1, parent_indices)
    translations = body._joint_offsets[1:] + translation_delta[..., 1:, :]
    translations = translations * parent_scales[..., :, None]
    quaternions = body._joint_pre_rotations[1:].expand(*translations.shape[:-1], 4)
    native_placements = torch.cat((translations, quaternions), dim=-1)

    base_placements = body._tree.robot.joint_placements
    joint_placements = base_placements.expand(
        *native_placements.shape[:-2],
        *base_placements.shape,
    )
    joint_placements = joint_placements.index_copy(
        -2,
        body._tree.native_origin_joint_indices,
        native_placements,
    )
    robot = body._tree.robot.with_values(joint_placements=joint_placements)
    neutral_data = br.forward_kinematics(robot, robot.q_neutral)
    native_poses = neutral_data.joint_pose_world.index_select(
        -2,
        body.structure.native_pose_joint_indices,
    )

    body_inertias = None
    if body.structure.part_topology is not None:
        transforms = skinning_transforms_from_inverse(native_poses, body.values.inverse_bind_poses)
        neutral_vertices = lbs(
            v_shaped,
            transforms,
            body.values.skinning_weight_matrix,
            joint_scales=joint_scales,
            bind_joint_positions=body.values.bind_joint_positions,
        )
        part_poses = native_poses.index_select(-2, body._part_native_joint_indices)
        compact_inertias = body_inertias_from_shape(
            neutral_vertices,
            part_poses,
            body.structure.part_topology,
            body.values.density,
        )
        target_indices = torch.cat(
            (
                body._part_body_indices.new_zeros(1),
                body._part_body_indices,
            )
        )
        body_inertias = compact_inertias.new_zeros(
            *compact_inertias.shape[:-2],
            robot.nbodies,
            10,
        ).index_copy(-2, target_indices, compact_inertias)
        robot = robot.with_values(body_inertias=body_inertias)

    values = replace(
        body.values,
        shape_parameters=expanded_parameters,
        v_shaped=v_shaped,
        rest_joints=native_poses[..., :3],
        joint_scales=joint_scales,
        body_inertias=body_inertias,
    )
    return values, robot


def mhr_pose_corrective(body: ParametricBody, data: Data) -> torch.Tensor:
    if body._corrective_first_layer_weight is None:
        return torch.zeros_like(body.values.v_shaped)
    q_full = expand_configuration(body.robot.structure, data.q)
    entries = q_full.index_select(-1, body._tree.corrective_full_q_indices)
    euler = entries @ body._tree.corrective_euler_transform.mT
    feature = pose_feature_6d_from_euler(euler.reshape(*euler.shape[:-1], 125, 3))
    return mlp_corrective(
        feature,
        body._corrective_first_layer_weight,
        body._corrective_output_weight,
    )
