"""Compose a converted MHR archive as a BetterHuman parametric body.

Archive validation, kinematic-tree construction, and deformation live in
focused sibling modules.  This module binds those products once and owns the
native 204-parameter classic conversion boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from better_robot.lie import se3, so3

from better_human.core import BodyStructure, BodyValues, ParametricBody
from better_human.core.classic import broadcast_configuration, classic_tensor
from better_human.core.mass import build_part_topology
from better_human.core.pipeline import dense_skinning_weights, dense_weight_from_sparse
from better_human.io import MHR_CORRECTIVE_FIELDS, MHR_SCHEMA, load_model_npz, resolve_model_path

from ..registry import register
from ..smpl_family.shared import density_tensor
from . import archive, deformation, tree
from .archive import EXPRESSION_COUNT, IDENTITY_COUNT, MODEL_PARAMETER_COUNT, ROOT_PARAMETER_COUNT
from .config import MHR_CONFIG


@dataclass(frozen=True)
class MHRClassic:
    """Native MHR inputs with arbitrary leading batch dimensions."""

    #: Identity blendshape coefficients with shape ``(..., 45)``.
    identity_coeffs: torch.Tensor
    #: Native compact model parameters with shape ``(..., 204)``.
    model_parameters: torch.Tensor
    #: Optional facial expression coefficients with shape ``(..., 72)``.
    expression: torch.Tensor | None = None


@register("mhr")
class MHR(ParametricBody):
    """Load one MHR LOD as a free-flyer plus 125 named revolute parameters.

    Six shared spine coordinates store their scaled first-row contribution in
    ``q`` (for example ``q[spine_twist0] = 0.01 * spine_twist0``). The classic
    converter applies and removes those archive-derived host weights.
    """

    def __init__(  # noqa: PLR0915 - one audited archive binding boundary
        self,
        model_path: str | Path | None = None,
        lod: int = MHR_CONFIG.default_lod,
        use_expression: bool = True,
        use_correctives: bool = True,
        density: float | torch.Tensor = 1000.0,
        compute_mass: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(lod, int) or isinstance(lod, bool) or lod not in MHR_CONFIG.lods:
            raise ValueError(f"lod must be one of {MHR_CONFIG.lods}, got {lod!r}")
        for name, value in (
            ("use_expression", use_expression),
            ("use_correctives", use_correctives),
            ("compute_mass", compute_mass),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError(f"dtype must be torch.float32 or torch.float64, got {dtype}")

        target_device = torch.device("cpu" if device is None else device)
        path = resolve_model_path(
            "MHR",
            MHR_CONFIG.asset_pattern.format(lod=lod),
            model_path,
        )
        excluded = set(MHR_CORRECTIVE_FIELDS) if not use_correctives else set()
        if not use_expression:
            excluded.add("expression_basis")
        arrays = load_model_npz(path, MHR_SCHEMA, exclude_fields=excluded)
        archive.validate_metadata(arrays, lod)
        tree_binding = tree.build_mhr_tree(arrays, dtype, target_device)

        v_template = archive.tensor(arrays, "template_vertices").to(device=target_device, dtype=dtype)
        faces = archive.tensor(arrays, "faces").to(device=target_device, dtype=torch.long)
        skinning_indices = archive.tensor(arrays, "skin_indices").to(device=target_device, dtype=torch.long)
        skinning_weights = archive.tensor(arrays, "skin_weights").to(device=target_device, dtype=dtype)
        identity_basis = archive.tensor(arrays, "identity_basis").to(device=target_device, dtype=dtype)
        expression_basis = (
            archive.tensor(arrays, "expression_basis").to(device=target_device, dtype=dtype)
            if use_expression
            else None
        )
        inverse_bind_poses = se3.from_matrix(
            archive.tensor(arrays, "inverse_bind_matrices").to(device=target_device, dtype=dtype)
        )
        bind_joint_positions = archive.tensor(arrays, "bind_matrices").to(
            device=target_device,
            dtype=dtype,
        )[..., :3, 3]

        parameter_groups: dict[str, int] = {"identity": IDENTITY_COUNT}
        if use_expression:
            parameter_groups["expression"] = EXPRESSION_COUNT
        parameter_groups["proportions"] = len(tree_binding.proportion_parameter_indices)
        parameters = {name: v_template.new_zeros(width) for name, width in parameter_groups.items()}
        part_ids, part_native_joint_indices = archive.segmentation(
            arrays,
            skinning_weights,
            skinning_indices,
        )
        part_topology = build_part_topology(faces, part_ids, len(part_native_joint_indices)) if compute_mass else None
        density_value = density_tensor(density, len(part_native_joint_indices), v_template)
        part_body_indices = torch.tensor(
            [
                tree_binding.robot.body_id(archive.strings(arrays, "joint_names")[int(index)])
                for index in part_native_joint_indices
            ],
            dtype=torch.long,
            device=target_device,
        )

        structure = BodyStructure(
            num_vertices=len(v_template),
            faces=faces,
            joint_names=archive.strings(arrays, "joint_names"),
            parents=tuple(int(value) for value in archive.tensor(arrays, "joint_parents").tolist()),
            native_joint_indices=tuple(range(len(tree_binding.native_pose_joint_indices))),
            skinning_indices=skinning_indices,
            part_ids=part_ids,
            part_topology=part_topology,
            parameter_groups=parameter_groups,
            landmarks={},
            unit=MHR_CONFIG.unit,
            up_axis=MHR_CONFIG.up_axis,
            forward_axis=MHR_CONFIG.forward_axis,
            native_pose_format=MHR_CONFIG.native_pose_format,
            native_pose_joint_indices=tree_binding.native_pose_joint_indices,
        )
        skinning_weight_matrix = dense_skinning_weights(
            skinning_weights,
            skinning_indices,
            len(structure.joint_names),
        )
        values = BodyValues(
            v_template=v_template,
            shape_basis=None,
            posedirs=None,
            joint_regressor=None,
            skinning_weights=skinning_weights,
            skinning_weight_matrix=skinning_weight_matrix,
            shape_parameters=parameters,
            v_shaped=v_template,
            rest_joints=v_template.new_zeros((len(structure.joint_names), 3)),
            pelvis_offset=None,
            density=density_value,
            body_inertias=None,
            inverse_bind_poses=inverse_bind_poses,
            bind_joint_positions=bind_joint_positions,
            joint_scales=v_template.new_ones(len(structure.joint_names)),
        )
        translation_basis, scale_basis, root_transform = archive.parameter_bases(
            arrays,
            tree_binding.proportion_parameter_indices,
            dtype,
            target_device,
        )

        first_layer_weight = None
        output_weight = None
        if use_correctives:
            first_layer_weight = dense_weight_from_sparse(
                archive.tensor(arrays, "corrective_first_layer_indices").to(
                    device=target_device,
                    dtype=torch.long,
                ),
                archive.tensor(arrays, "corrective_first_layer_weights").to(
                    device=target_device,
                    dtype=dtype,
                ),
                archive.tensor(arrays, "corrective_first_layer_mask").to(
                    device=target_device,
                    dtype=torch.bool,
                ),
            )
            output_weight = archive.tensor(arrays, "corrective_output_weights").to(
                device=target_device,
                dtype=dtype,
            )

        object.__setattr__(self, "structure", structure)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "robot", tree_binding.robot)
        object.__setattr__(self, "_shape_bake_policy", deformation.bake_mhr_shape)
        object.__setattr__(self, "_pose_corrective_policy", deformation.mhr_pose_corrective)
        object.__setattr__(self, "_tree", tree_binding)
        object.__setattr__(self, "_identity_basis", identity_basis)
        object.__setattr__(self, "_expression_basis", expression_basis)
        object.__setattr__(self, "_use_expression", use_expression)
        object.__setattr__(
            self,
            "_joint_offsets",
            archive.tensor(arrays, "joint_translation_offsets").to(target_device, dtype),
        )
        object.__setattr__(
            self,
            "_joint_pre_rotations",
            so3.normalize(archive.tensor(arrays, "joint_pre_rotations").to(target_device, dtype)),
        )
        object.__setattr__(self, "_translation_basis", translation_basis)
        object.__setattr__(self, "_scale_basis", scale_basis)
        object.__setattr__(self, "_root_parameter_transform", root_transform)
        object.__setattr__(self, "_root_parameter_transform_inverse", torch.linalg.inv(root_transform))
        object.__setattr__(self, "_part_native_joint_indices", part_native_joint_indices)
        object.__setattr__(self, "_part_body_indices", part_body_indices)
        object.__setattr__(self, "_corrective_first_layer_weight", first_layer_weight)
        object.__setattr__(self, "_corrective_output_weight", output_weight)

        baked_values, baked_robot = deformation.bake_mhr_shape(self, parameters)
        object.__setattr__(self, "values", baked_values)
        object.__setattr__(self, "robot", baked_robot)

    def from_classic(self, classic: MHRClassic) -> tuple[MHR, torch.Tensor]:
        """Bake classic shape state and return ``(shaped_body, q)``.

        ``expression=None`` is the reference's zero-expression shorthand and
        is canonicalized to an explicit zero tensor when expression is enabled.
        """
        exemplar = self.values.v_template
        identity = classic_tensor("identity_coeffs", classic.identity_coeffs, IDENTITY_COUNT, exemplar)
        model_parameters = classic_tensor(
            "model_parameters",
            classic.model_parameters,
            MODEL_PARAMETER_COUNT,
            exemplar,
        )
        if identity is None or model_parameters is None:
            raise TypeError("identity_coeffs and model_parameters must be floating torch.Tensors")
        if not self._use_expression and classic.expression is not None:
            raise ValueError("expression requires an MHR body constructed with use_expression=True")
        expression = (
            classic_tensor("expression", classic.expression, EXPRESSION_COUNT, exemplar)
            if self._use_expression
            else None
        )
        shapes = [identity.shape[:-1], model_parameters.shape[:-1]]
        if expression is not None:
            shapes.append(expression.shape[:-1])
        batch_shape = torch.broadcast_shapes(*shapes)
        identity = identity.expand(*batch_shape, IDENTITY_COUNT)
        model_parameters = model_parameters.expand(*batch_shape, MODEL_PARAMETER_COUNT)
        shape_groups = {
            "identity": identity,
            "proportions": model_parameters.index_select(-1, self._tree.proportion_parameter_indices),
        }
        if self._use_expression:
            shape_groups["expression"] = (
                exemplar.new_zeros((*batch_shape, EXPRESSION_COUNT))
                if expression is None
                else expression.expand(*batch_shape, EXPRESSION_COUNT)
            )
        shaped = self.with_shape(**shape_groups)

        root_parameters = model_parameters[..., :ROOT_PARAMETER_COUNT]
        root_channels = root_parameters @ self._root_parameter_transform.mT
        root = torch.cat((root_channels[..., :3], so3.from_euler(root_channels[..., 3:6])), dim=-1)
        pose = model_parameters.index_select(-1, self._tree.pose_parameter_indices) * self._tree.pose_host_weights
        external_q = torch.cat((root, pose), dim=-1)
        return shaped, external_q.index_select(-1, self._tree.q_from_classic_permutation)

    def to_classic(self, q: torch.Tensor) -> MHRClassic:
        """Convert ``q`` plus this body's baked shape back to native MHR.

        Euler angles use BetterRobot's principal branch away from gimbal lock.
        A ``2*pi`` roll representative preserves a negative root-quaternion
        sign so the vector-valued ``q`` round trip remains lossless.
        """
        identity = self.values.shape_parameters["identity"]
        q, batch_shape = broadcast_configuration(q, self.robot, identity)
        external_q = q.index_select(-1, self._tree.q_to_classic_permutation)
        root_quaternion = so3.normalize(external_q[..., 3:7])
        root_euler = so3.to_euler(root_quaternion)
        reconstructed = so3.from_euler(root_euler)
        needs_sign_flip = (reconstructed * root_quaternion).sum(dim=-1, keepdim=True) < 0
        roll = root_euler[..., :1] + torch.where(
            needs_sign_flip,
            root_euler.new_tensor(2.0 * torch.pi),
            root_euler.new_tensor(0.0),
        )
        root_euler = torch.cat((roll, root_euler[..., 1:]), dim=-1)
        root_channels = torch.cat((external_q[..., :3], root_euler), dim=-1)
        root_parameters = root_channels @ self._root_parameter_transform_inverse.mT
        pose_parameters = external_q[..., 7:] / self._tree.pose_host_weights
        proportions = self.values.shape_parameters["proportions"].expand(*batch_shape, 73)

        model_parameters = q.new_zeros((*batch_shape, MODEL_PARAMETER_COUNT))
        root_indices = torch.arange(ROOT_PARAMETER_COUNT, dtype=torch.long, device=q.device)
        model_parameters = model_parameters.index_copy(-1, root_indices, root_parameters)
        model_parameters = model_parameters.index_copy(-1, self._tree.pose_parameter_indices, pose_parameters)
        model_parameters = model_parameters.index_copy(
            -1,
            self._tree.proportion_parameter_indices,
            proportions,
        )
        expression = (
            self.values.shape_parameters["expression"].expand(*batch_shape, EXPRESSION_COUNT)
            if self._use_expression
            else None
        )
        return MHRClassic(
            identity_coeffs=identity.expand(*batch_shape, IDENTITY_COUNT),
            model_parameters=model_parameters,
            expression=expression,
        )


__all__ = ["MHR", "MHRClassic"]
