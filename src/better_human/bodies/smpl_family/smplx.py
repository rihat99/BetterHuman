"""SMPL-X data binding, partial spawning, and classic conversion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from better_human.core import BodyStructure, ParametricBody
from better_human.core.classic import (
    axis_angle_to_quaternion,
    broadcast_configuration,
    classic_tensor,
    classic_root_to_free_flyer,
    free_flyer_to_classic_root,
    quaternion_to_axis_angle,
)
from better_human.core.reduction import reduce_joint_data, slice_parameter_basis
from better_human.io import SMPLX_SCHEMA, load_model_npz, resolve_model_path

from ..assets import load_density_table
from ..registry import register
from .config_smplx import SMPLX_CONFIG
from .shared import (
    bind_mass_properties,
    bind_smpl_body,
    build_family_robot,
    contact_frame_ir,
    density_tensor,
    make_body_values,
    parents_from_kintree,
)

SHAPE_SPACE_DIM = 300
EXPRESSION_SPACE_DIM = 100
FULL_HAND_DIM = 45


@dataclass(frozen=True)
class SMPLXClassic:
    """Classic SMPL-X parameters with arbitrary leading batch dimensions."""

    #: Root orientation as axis angle with shape ``(..., 3)``.
    global_orient: torch.Tensor | None = None
    #: The 21 body-joint axis angles with shape ``(..., 63)``.
    body_pose: torch.Tensor | None = None
    #: Jaw axis angle with shape ``(..., 3)``.
    jaw_pose: torch.Tensor | None = None
    #: Left-eye axis angle with shape ``(..., 3)``.
    leye_pose: torch.Tensor | None = None
    #: Right-eye axis angle with shape ``(..., 3)``.
    reye_pose: torch.Tensor | None = None
    #: Full 45-D deltas or ``num_pca_comps`` coefficients for the left hand.
    left_hand_pose: torch.Tensor | None = None
    #: Full 45-D deltas or ``num_pca_comps`` coefficients for the right hand.
    right_hand_pose: torch.Tensor | None = None
    #: Baked expression coefficients with shape ``(..., num_expression)``.
    expression: torch.Tensor | None = None
    #: Post-skinning world translation with shape ``(..., 3)``.
    transl: torch.Tensor | None = None
    #: Hand PCA width, or ``None`` when hand poses are full axis-angle deltas.
    num_pca_comps: int | None = None


def _pca_width(num_pca_comps: int | None) -> int:
    if num_pca_comps is None:
        return FULL_HAND_DIM
    if (
        not isinstance(num_pca_comps, int)
        or isinstance(num_pca_comps, bool)
        or num_pca_comps < 1
        or num_pca_comps > FULL_HAND_DIM
    ):
        raise ValueError(f"num_pca_comps must be in [1, {FULL_HAND_DIM}] or None")
    return num_pca_comps


def _expand_or_zeros(
    value: torch.Tensor | None,
    batch_shape: tuple[int, ...],
    width: int,
    exemplar: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        return exemplar.new_zeros((*batch_shape, width))
    return value.expand(*batch_shape, width)


@register("smplx")
class SMPLX(ParametricBody):
    """Load SMPL-X with optional terminal hand and face subtree reduction."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        gender: str = "neutral",
        num_betas: int = 16,
        num_expression: int = 10,
        use_hands: bool = True,
        use_face: bool = True,
        flat_hand_mean: bool = False,
        density: float | torch.Tensor | str = 1000.0,
        compute_mass: bool = True,
        contact_frames: str | Path | Mapping[str, tuple[int, str | None]] | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        normalized_gender = gender.lower()
        if normalized_gender not in {"female", "male", "neutral"}:
            raise ValueError("gender must be 'female', 'male', or 'neutral'")
        if not dtype.is_floating_point:
            raise TypeError(f"dtype must be floating, got {dtype}")
        if num_betas < 1 or num_betas > SHAPE_SPACE_DIM:
            raise ValueError(f"num_betas must be in [1, {SHAPE_SPACE_DIM}], got {num_betas}")
        if use_face and (num_expression < 1 or num_expression > EXPRESSION_SPACE_DIM):
            raise ValueError(
                f"num_expression must be in [1, {EXPRESSION_SPACE_DIM}] when use_face=True, got {num_expression}"
            )

        target_device = torch.device("cpu" if device is None else device)
        path = resolve_model_path(
            "smplx",
            f"SMPLX_{normalized_gender.upper()}.npz",
            model_path,
        )
        arrays = load_model_npz(path, SMPLX_SCHEMA)
        joint_count = arrays["weights"].shape[-1]
        if joint_count != len(SMPLX_CONFIG.joint_names):
            raise ValueError(f"SMPL-X weights contain {joint_count} joints; expected {len(SMPLX_CONFIG.joint_names)}")

        parents = parents_from_kintree(arrays["kintree_table"])
        requested_groups = {"betas": num_betas}
        if use_face:
            requested_groups["expression"] = num_expression
        shape_basis, parameter_groups = slice_parameter_basis(
            arrays["shapedirs"],
            {
                "betas": slice(0, SHAPE_SPACE_DIM),
                "expression": slice(SHAPE_SPACE_DIM, SHAPE_SPACE_DIM + EXPRESSION_SPACE_DIM),
            },
            requested_groups,
        )

        dropped: list[int] = []
        if not use_face:
            dropped.extend(SMPLX_CONFIG.face_joints)
        if not use_hands:
            dropped.extend(SMPLX_CONFIG.left_hand_joints + SMPLX_CONFIG.right_hand_joints)
        reduced = reduce_joint_data(
            SMPLX_CONFIG.joint_names,
            parents,
            arrays["J_regressor"],
            arrays["weights"],
            arrays["posedirs"],
            dropped,
        )
        joint_names, reduced_parents, regressor, weights, posedirs, survivors = reduced
        v_template = arrays["v_template"].to(device=target_device, dtype=dtype)
        density_value = density_tensor(
            load_density_table(SMPLX_CONFIG.density_table, density, joint_names, v_template)
            if isinstance(density, str)
            else density,
            len(joint_names),
            v_template,
        )
        values, skinning_indices = make_body_values(
            v_template,
            shape_basis.to(device=target_device, dtype=dtype),
            posedirs.to(device=target_device, dtype=dtype),
            regressor.to(device=target_device, dtype=dtype),
            weights.to(device=target_device, dtype=dtype),
            parameter_groups,
            density_value,
        )
        faces = arrays["f"].to(device=target_device, dtype=torch.long)
        values, part_ids, part_topology = bind_mass_properties(
            values,
            faces,
            skinning_indices,
            len(joint_names),
            compute_mass,
        )
        structure = BodyStructure(
            num_vertices=arrays["v_template"].shape[0],
            faces=faces,
            joint_names=joint_names,
            parents=reduced_parents,
            native_joint_indices=survivors,
            skinning_indices=skinning_indices,
            part_ids=part_ids,
            part_topology=part_topology,
            parameter_groups=parameter_groups,
            landmarks={
                "frame_vertex_ids": torch.tensor(
                    SMPLX_CONFIG.frame_vertex_ids, dtype=torch.long, device=target_device
                )
            },
            unit=SMPLX_CONFIG.unit,
            up_axis=SMPLX_CONFIG.up_axis,
            forward_axis=SMPLX_CONFIG.forward_axis,
            native_pose_format=SMPLX_CONFIG.native_pose_format,
        )
        frame_specs, contact_frame_spec = (
            (None, None) if contact_frames is None else contact_frame_ir(contact_frames, structure, values)
        )
        robot = build_family_robot(
            f"smplx_{normalized_gender}",
            joint_names,
            reduced_parents,
            values.rest_joints,
            dtype,
            target_device,
            frame_specs,
        )
        if values.body_inertias is not None:
            robot = robot.with_values(body_inertias=values.body_inertias)

        left_mean = arrays["hands_meanl"].to(device=target_device, dtype=dtype)
        right_mean = arrays["hands_meanr"].to(device=target_device, dtype=dtype)
        if flat_hand_mean:
            left_mean = torch.zeros_like(left_mean)
            right_mean = torch.zeros_like(right_mean)
        bind_smpl_body(self, structure, values, robot)
        object.__setattr__(self, "contact_frames", contact_frame_spec)
        object.__setattr__(
            self,
            "_left_hand_components",
            arrays["hands_componentsl"].to(device=target_device, dtype=dtype),
        )
        object.__setattr__(
            self,
            "_right_hand_components",
            arrays["hands_componentsr"].to(device=target_device, dtype=dtype),
        )
        object.__setattr__(self, "_left_hand_mean", left_mean)
        object.__setattr__(self, "_right_hand_mean", right_mean)
        object.__setattr__(self, "_use_hands", use_hands)
        object.__setattr__(self, "_use_face", use_face)

    def from_classic(self, classic: SMPLXClassic) -> torch.Tensor:
        """Convert classic SMPL-X parameters to BetterRobot ``q (..., nq)``."""
        config = SMPLX_CONFIG
        pelvis = self.values.pelvis_offset
        global_orient = classic_tensor("global_orient", classic.global_orient, 3, pelvis)
        body_pose = classic_tensor("body_pose", classic.body_pose, len(config.body_joints) * 3, pelvis)
        transl = classic_tensor("transl", classic.transl, 3, pelvis)

        if not self._use_face and any(
            value is not None for value in (classic.jaw_pose, classic.leye_pose, classic.reye_pose, classic.expression)
        ):
            raise ValueError("Face parameters require an SMPLX body constructed with use_face=True")
        jaw_pose = classic_tensor("jaw_pose", classic.jaw_pose, 3, pelvis) if self._use_face else None
        leye_pose = classic_tensor("leye_pose", classic.leye_pose, 3, pelvis) if self._use_face else None
        reye_pose = classic_tensor("reye_pose", classic.reye_pose, 3, pelvis) if self._use_face else None
        expression_width = self.structure.parameter_groups.get("expression")
        expression = (
            classic_tensor("expression", classic.expression, expression_width, pelvis)
            if expression_width is not None
            else None
        )

        if not self._use_hands and (
            classic.left_hand_pose is not None
            or classic.right_hand_pose is not None
            or classic.num_pca_comps is not None
        ):
            raise ValueError("Hand parameters require an SMPLX body constructed with use_hands=True")
        hand_width = _pca_width(classic.num_pca_comps) if self._use_hands else FULL_HAND_DIM
        left_hand_pose = (
            classic_tensor("left_hand_pose", classic.left_hand_pose, hand_width, pelvis)
            if self._use_hands
            else None
        )
        right_hand_pose = (
            classic_tensor("right_hand_pose", classic.right_hand_pose, hand_width, pelvis)
            if self._use_hands
            else None
        )

        values = [global_orient, body_pose, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose, expression, transl]
        batch_shapes = [pelvis.shape[:-1]]
        batch_shapes.extend(value.shape[:-1] for value in values if value is not None)
        batch_shape = torch.broadcast_shapes(*batch_shapes)
        global_orient = _expand_or_zeros(global_orient, batch_shape, 3, pelvis)
        body_pose = _expand_or_zeros(body_pose, batch_shape, len(config.body_joints) * 3, pelvis)
        transl = _expand_or_zeros(transl, batch_shape, 3, pelvis)
        pelvis = pelvis.expand(*batch_shape, 3)

        if expression is not None:
            expression = expression.expand(*batch_shape, expression_width)
            baked_expression = self.values.shape_parameters["expression"].expand(*batch_shape, expression_width)
            if not torch.equal(expression, baked_expression):
                raise ValueError(
                    "SMPLXClassic.expression is shape state; call body.with_shape(expression=...) "
                    "with the same tensor before from_classic"
                )

        face_axes = torch.cat(
            (
                _expand_or_zeros(jaw_pose, batch_shape, 3, pelvis),
                _expand_or_zeros(leye_pose, batch_shape, 3, pelvis),
                _expand_or_zeros(reye_pose, batch_shape, 3, pelvis),
            ),
            dim=-1,
        ).reshape(*batch_shape, len(config.face_joints), 3)

        if self._use_hands:
            left_delta = _expand_or_zeros(left_hand_pose, batch_shape, hand_width, pelvis)
            right_delta = _expand_or_zeros(right_hand_pose, batch_shape, hand_width, pelvis)
            if classic.num_pca_comps is not None:
                left_delta = left_delta @ self._left_hand_components[:hand_width]
                right_delta = right_delta @ self._right_hand_components[:hand_width]
            left_axes = (left_delta + self._left_hand_mean).reshape(*batch_shape, len(config.left_hand_joints), 3)
            right_axes = (right_delta + self._right_hand_mean).reshape(*batch_shape, len(config.right_hand_joints), 3)
        else:
            left_axes = pelvis.new_zeros((*batch_shape, len(config.left_hand_joints), 3))
            right_axes = pelvis.new_zeros((*batch_shape, len(config.right_hand_joints), 3))

        native_nonroot_axes = torch.cat(
            (
                body_pose.reshape(*batch_shape, len(config.body_joints), 3),
                face_axes,
                left_axes,
                right_axes,
            ),
            dim=-2,
        )
        gather = torch.tensor(
            [native_index - 1 for native_index in self.structure.native_joint_indices[1:]],
            dtype=torch.long,
            device=pelvis.device,
        )
        reduced_axes = native_nonroot_axes.index_select(-2, gather)
        root = classic_root_to_free_flyer(global_orient, transl, pelvis)
        return torch.cat((root, axis_angle_to_quaternion(reduced_axes).flatten(start_dim=-2)), dim=-1)

    def to_classic(self, q: torch.Tensor, num_pca_comps: int | None = None) -> SMPLXClassic:
        """Convert BetterRobot ``q`` to full or PCA-hand classic parameters."""
        config = SMPLX_CONFIG
        if not self._use_hands and num_pca_comps is not None:
            raise ValueError("num_pca_comps requires an SMPLX body constructed with use_hands=True")
        hand_width = _pca_width(num_pca_comps) if self._use_hands else FULL_HAND_DIM
        q, batch_shape = broadcast_configuration(q, self.robot, self.values.pelvis_offset)
        pelvis = self.values.pelvis_offset.expand(*batch_shape, 3)
        root_width = self.robot.nqs[1]
        global_orient, transl = free_flyer_to_classic_root(q[..., :root_width], pelvis)
        reduced_axes = quaternion_to_axis_angle(q[..., root_width:].reshape(*batch_shape, -1, 4))
        reduced_by_native = {
            native: reduced
            for reduced, native in enumerate(self.structure.native_joint_indices[1:])
        }

        def gather(native_indices: tuple[int, ...]) -> torch.Tensor:
            positions = torch.tensor(
                [reduced_by_native[index] for index in native_indices],
                dtype=torch.long,
                device=q.device,
            )
            return reduced_axes.index_select(-2, positions)

        body_pose = gather(config.body_joints).flatten(start_dim=-2)
        if self._use_face:
            face_axes = gather(config.face_joints)
            jaw_pose, leye_pose, reye_pose = face_axes.unbind(dim=-2)
            expression_width = self.structure.parameter_groups["expression"]
            expression = self.values.shape_parameters["expression"].expand(*batch_shape, expression_width)
        else:
            jaw_pose = leye_pose = reye_pose = expression = None

        if self._use_hands:
            left_delta = gather(config.left_hand_joints).flatten(start_dim=-2) - self._left_hand_mean
            right_delta = gather(config.right_hand_joints).flatten(start_dim=-2) - self._right_hand_mean
            if num_pca_comps is None:
                left_hand_pose = left_delta
                right_hand_pose = right_delta
            else:
                left_components = self._left_hand_components[:hand_width]
                right_components = self._right_hand_components[:hand_width]
                left_hand_pose = left_delta @ torch.linalg.pinv(left_components)
                right_hand_pose = right_delta @ torch.linalg.pinv(right_components)
                if not torch.allclose(left_hand_pose @ left_components, left_delta, rtol=2e-5, atol=2e-5):
                    raise ValueError(f"left hand pose is outside the first {hand_width} PCA components")
                if not torch.allclose(right_hand_pose @ right_components, right_delta, rtol=2e-5, atol=2e-5):
                    raise ValueError(f"right hand pose is outside the first {hand_width} PCA components")
        else:
            left_hand_pose = right_hand_pose = None

        return SMPLXClassic(
            global_orient=global_orient,
            body_pose=body_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            expression=expression,
            transl=transl,
            num_pca_comps=num_pca_comps,
        )


__all__ = ["SMPLX", "SMPLXClassic"]
