"""SMPL data binding, classic conversion, and BetterRobot tree policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from better_human.core import BodyStructure, ParametricBody
from better_human.core.classic import (
    axis_angle_to_quaternion,
    classic_root_to_free_flyer,
    free_flyer_to_classic_root,
    quaternion_to_axis_angle,
)
from better_human.core.reduction import slice_parameter_basis
from better_human.io import SMPL_SCHEMA, load_model_npz, resolve_model_path

from .assets import load_closed_segmentation, load_density_table
from .config_smpl import SMPL_CONFIG
from .registry import register
from .smpl_family import (
    bind_mass_properties,
    broadcast_configuration,
    build_family_robot,
    classic_tensor,
    density_tensor,
    make_body_values,
    parents_from_kintree,
)


@dataclass(frozen=True)
class SMPLClassic:
    """Classic SMPL parameters with arbitrary leading batch dimensions."""

    #: Root orientation as an axis-angle tensor with shape ``(..., 3)``.
    global_orient: torch.Tensor | None = None
    #: Concatenated non-root axis angles with shape ``(..., 69)``.
    body_pose: torch.Tensor | None = None
    #: Post-skinning world translation with shape ``(..., 3)``.
    transl: torch.Tensor | None = None


@register("smpl")
class SMPL(ParametricBody):
    """Load an SMPL body and bind its native joint order to BetterRobot."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        gender: str = "neutral",
        num_betas: int = 10,
        density: float | torch.Tensor | str = 1000.0,
        compute_mass: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        normalized_gender = gender.lower()
        if normalized_gender not in {"female", "male", "neutral"}:
            raise ValueError("gender must be 'female', 'male', or 'neutral'")
        if not dtype.is_floating_point:
            raise TypeError(f"dtype must be floating, got {dtype}")
        target_device = torch.device("cpu" if device is None else device)
        path = resolve_model_path(
            "smpl",
            f"SMPL_{normalized_gender.upper()}.npz",
            model_path,
        )
        arrays = load_model_npz(path, SMPL_SCHEMA)
        available_betas = arrays["shapedirs"].shape[-1]
        if num_betas < 1 or num_betas > available_betas:
            raise ValueError(f"num_betas must be in [1, {available_betas}], got {num_betas}")

        joint_names = SMPL_CONFIG.joint_names
        joint_count = arrays["weights"].shape[-1]
        if joint_count != len(joint_names):
            raise ValueError(f"SMPL weights contain {joint_count} joints; expected {len(joint_names)}")
        parents = parents_from_kintree(arrays["kintree_table"])
        shape_basis, parameter_groups = slice_parameter_basis(
            arrays["shapedirs"],
            {"betas": slice(0, available_betas)},
            {"betas": num_betas},
        )
        v_template = arrays["v_template"].to(device=target_device, dtype=dtype)
        if isinstance(density, str):
            density = load_density_table(SMPL_CONFIG.density_table, density, joint_count, v_template)
        density_value = density_tensor(density, joint_count, v_template)
        values, skinning_indices = make_body_values(
            v_template,
            shape_basis.to(device=target_device, dtype=dtype),
            arrays["posedirs"].to(device=target_device, dtype=dtype),
            arrays["J_regressor"].to(device=target_device, dtype=dtype),
            arrays["weights"].to(device=target_device, dtype=dtype),
            parameter_groups,
            density_value,
        )
        faces = arrays["f"].to(device=target_device, dtype=torch.long)
        closed_faces = None
        if compute_mass and SMPL_CONFIG.segmentation_closed_file is not None:
            closed_faces = load_closed_segmentation(SMPL_CONFIG.segmentation_closed_file, target_device)
        values, part_ids, part_topology = bind_mass_properties(
            values,
            faces,
            skinning_indices,
            joint_count,
            compute_mass,
            closed_faces,
        )
        structure = BodyStructure(
            num_vertices=arrays["v_template"].shape[0],
            faces=faces,
            joint_names=joint_names,
            parents=parents,
            native_joint_indices=tuple(range(joint_count)),
            skinning_indices=skinning_indices,
            part_ids=part_ids,
            part_topology=part_topology,
            parameter_groups=parameter_groups,
            landmarks={
                "frame_vertex_ids": torch.tensor(
                    SMPL_CONFIG.frame_vertex_ids, dtype=torch.long, device=target_device
                )
            },
            unit=SMPL_CONFIG.unit,
            up_axis=SMPL_CONFIG.up_axis,
            forward_axis=SMPL_CONFIG.forward_axis,
            native_pose_format=SMPL_CONFIG.native_pose_format,
        )

        robot = build_family_robot(
            f"smpl_{normalized_gender}",
            joint_names,
            parents,
            values.rest_joints,
            dtype,
            target_device,
        )
        if values.body_inertias is not None:
            robot = robot.with_values(body_inertias=values.body_inertias)
        object.__setattr__(self, "structure", structure)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "robot", robot)

    def from_classic(self, classic: SMPLClassic) -> torch.Tensor:
        """Convert classic SMPL parameters to BetterRobot ``q (..., nq)``."""
        pelvis = self.values.pelvis_offset
        body_width = (len(self.structure.joint_names) - 1) * 3
        global_orient = classic_tensor("global_orient", classic.global_orient, 3, pelvis)
        body_pose = classic_tensor("body_pose", classic.body_pose, body_width, pelvis)
        transl = classic_tensor("transl", classic.transl, 3, pelvis)
        batch_shapes = [pelvis.shape[:-1]]
        batch_shapes.extend(value.shape[:-1] for value in (global_orient, body_pose, transl) if value is not None)
        batch_shape = torch.broadcast_shapes(*batch_shapes)
        if global_orient is None:
            global_orient = pelvis.new_zeros((*batch_shape, 3))
        else:
            global_orient = global_orient.expand(*batch_shape, 3)
        if body_pose is None:
            body_pose = pelvis.new_zeros((*batch_shape, body_width))
        else:
            body_pose = body_pose.expand(*batch_shape, body_width)
        if transl is None:
            transl = pelvis.new_zeros((*batch_shape, 3))
        else:
            transl = transl.expand(*batch_shape, 3)
        pelvis = pelvis.expand(*batch_shape, 3)

        root = classic_root_to_free_flyer(global_orient, transl, pelvis)
        body_quaternions = axis_angle_to_quaternion(body_pose.reshape(*batch_shape, -1, 3))
        return torch.cat((root, body_quaternions.flatten(start_dim=-2)), dim=-1)

    def to_classic(self, q: torch.Tensor) -> SMPLClassic:
        """Convert BetterRobot ``q (..., nq)`` to classic SMPL parameters."""
        q, batch_shape = broadcast_configuration(q, self.robot, self.values.pelvis_offset)
        pelvis = self.values.pelvis_offset.expand(*batch_shape, 3)
        root_width = self.robot.nqs[1]
        global_orient, transl = free_flyer_to_classic_root(q[..., :root_width], pelvis)
        body_quaternions = q[..., root_width:].reshape(*batch_shape, -1, 4)
        body_pose = quaternion_to_axis_angle(body_quaternions).flatten(start_dim=-2)
        return SMPLClassic(global_orient=global_orient, body_pose=body_pose, transl=transl)


__all__ = ["SMPL", "SMPLClassic"]
