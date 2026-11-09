"""MHR rig: the authored joint tree as BR revolute chains, proportion-driven placements, Euler corrective features.

Mirrors the Momentum Human Rig (pymomentum). Native joints are the archive's 127 joints: ``body_world`` (the BR
universe), ``root`` (a free flyer), then joints whose local rotation is ``pre_rotation Rz(z) Ry(y) Rx(x)``. Each of
those is a chain of BR revolutes, one per parameter-transform entry of its rotation rows, about z, then y, then x; a
joint without entries is fixed. Each pose parameter drives one host revolute (its first entry of weight 1, else its
first entry); its other entries mimic the host.

``q (..., 132)`` is the root translation and quaternion, then the 125 host angles in BR joint order, each its pose
parameter times the host's weight. The 73 proportion parameters ``p`` place every joint at
``(offset + translation_basis p) s_parent`` and scale it by ``s = s_parent exp2(scale_basis p)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import torch
from better_robot import Model, ModelBuilder
from better_robot.io import build_model
from better_robot.kinematics import forward_kinematics, full_configuration
from better_robot.lie import se3, so3

from better_human.core.pipeline import pose_feature_6d_from_euler
from better_human.core.rig import Rig, RigBake, gravity

if TYPE_CHECKING:
    from better_human.core.spec import FamilySpec

ROOT_COUNT = 6  #: model parameters 0-5 drive the root: translation, then Euler angles.
CHANNELS = 7  #: parameter-transform rows per joint: translation xyz, rotation xyz, log2 scale.


@dataclass(frozen=True)
class MhrRig(Rig):
    """MHR rig tensors. ``J = 127`` native joints, ``N = 125`` pose and ``P = 73`` proportion parameters."""

    model: Model  #: float64 BR model at zero proportions; ``bake`` re-places its ``origin_joints``.
    native_joints: torch.Tensor  #: ``(J,)`` BR joint whose child body is each native joint (``body_world``: 0).
    origin_joints: torch.Tensor  #: ``(J - 1,)`` first BR joint of every native joint after ``body_world``.
    parents: torch.Tensor  #: ``(J - 1,)`` native parent of every native joint after ``body_world``.
    offsets: torch.Tensor  #: ``(J, 3)`` joint offsets in the parent frame, metres.
    pre_rotations: torch.Tensor  #: ``(J, 4)`` xyzw.
    translation_basis: torch.Tensor  #: ``(J, 3, P)`` offset change per proportion parameter.
    scale_basis: torch.Tensor  #: ``(J, P)`` local log2 scale per proportion parameter.
    inverse_bind_poses: torch.Tensor  #: ``(J, 7)`` authored inverse bind poses.
    bind_joint_positions: torch.Tensor  #: ``(J, 3)`` bind pivots of the accumulated joint scales.
    root_transform: torch.Tensor  #: ``(6, 6)`` root channels (translation, Euler xyz) from the root parameters.
    pose_columns: torch.Tensor  #: ``(N,)`` model parameter of every host angle ``q[..., 7:]``.
    host_weights: torch.Tensor  #: ``(N,)`` host angle per unit of its model parameter.
    proportion_columns: torch.Tensor  #: ``(P,)`` model parameters of the proportions.
    corrective_entries: torch.Tensor  #: ``(E,)`` full-configuration slot of every rotation entry of joints 2...
    corrective_rows: torch.Tensor  #: ``(E, 3 (J - 2))`` one-hot Euler angle (x, y, z of joints 2..) of every entry.

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, Any], *, dtype: torch.dtype, device: torch.device) -> MhrRig:
        """Rig from validated ``mhr_lod*.npz`` arrays (``tools/convert/mhr``)."""
        if any(parent >= joint for joint, parent in enumerate(arrays["joint_parents"].tolist())):
            raise ValueError("MHR joint parents must precede their children")
        offsets = arrays["joint_translation_offsets"].to(dtype)
        pre_rotations = so3.normalize(arrays["joint_pre_rotations"].to(dtype))
        if not torch.equal(pre_rotations[1], pre_rotations.new_tensor((0.0, 0.0, 0.0, 1.0))):
            raise ValueError("MHR root pre-rotation must be the identity: q[:3] is the root's world translation")
        kinds = arrays["model_parameter_channel_types"].tolist()
        proportions = [column for column, kind in enumerate(kinds) if kind in (0, 2) and column >= ROOT_COUNT]
        rows, columns = arrays["parameter_transform_rows"], arrays["parameter_transform_columns"]
        transform = torch.zeros(len(offsets) * CHANNELS, len(kinds), dtype=dtype)
        transform.index_put_((rows, columns), arrays["parameter_transform_values"].to(dtype))
        transform = transform.unflatten(0, (len(offsets), CHANNELS))
        tensors = {
            **_joint_tree(arrays, offsets, pre_rotations, dtype),
            "offsets": offsets,
            "pre_rotations": pre_rotations,
            "translation_basis": transform[:, :3, proportions],
            "scale_basis": transform[:, 6, proportions],
            "inverse_bind_poses": se3.from_matrix(arrays["inverse_bind_matrices"].to(dtype)),
            "bind_joint_positions": arrays["bind_matrices"].to(dtype)[..., :3, 3],
            "root_transform": transform[1, :ROOT_COUNT, :ROOT_COUNT].clone(),
            "parents": arrays["joint_parents"][1:],
            "proportion_columns": torch.tensor(proportions),
        }
        return cls(**tensors).to(dtype, device)

    def build_model(self, spec: FamilySpec, *, frames: Sequence[tuple[str, str, torch.Tensor]]) -> Model:
        """BR model at zero proportions in the spec's dtype; ``bake`` re-places it per identity."""
        if frames:
            raise ValueError("MHR does not support contact frames")
        return self.model.to(device=spec.device, dtype=spec.dtype)

    def bake(
        self, spec: FamilySpec, model: Model, *, shaped_vertices: torch.Tensor, identity: Mapping[str, torch.Tensor]
    ) -> RigBake:
        """Placements and scaled inverse binds for the proportions ``identity["proportions"] (I..., P)``.

        The accumulated joint scales ``s`` fold into the inverse binds ``[s R_b | t_b - (s - 1) R_b c]`` (bind rotation
        ``R_b``, translation ``t_b``, pivot ``c``), so skinning is one affine per joint (decision record 0003).
        """
        proportions = identity["proportions"]
        delta = torch.einsum("...p,jcp->...jc", proportions, self.translation_basis)
        scales = self.joint_scales(spec, proportions)
        translations = (self.offsets[1:] + delta[..., 1:, :]) * scales.index_select(-1, self.parents)[..., None]
        rotations = self.pre_rotations[1:].expand(*translations.shape[:-1], 4)
        base = model.joint_placements
        placements = base.expand(*translations.shape[:-2], *base.shape).index_copy(
            -2, self.origin_joints, torch.cat((translations, rotations), dim=-1)
        )
        baked = model.with_values(joint_placements=placements)
        kinematics = forward_kinematics(baked, baked.q_neutral)
        bind_rotations = so3.to_matrix(self.inverse_bind_poses[..., 3:])
        pivots = (bind_rotations @ self.bind_joint_positions[..., None]).squeeze(-1)
        bind_translations = self.inverse_bind_poses[..., :3] - (scales - 1.0)[..., None] * pivots
        return RigBake(
            joint_placements=placements,
            rest_vertices=shaped_vertices,
            rest_joint_translations=self.joint_transforms(spec, baked, kinematics=kinematics)[..., 3],
            inverse_bind_affines=torch.cat(
                (scales[..., None, None] * bind_rotations, bind_translations[..., None]), -1
            ),
            root_offset=placements.new_zeros((*placements.shape[:-2], 3)),
        )

    def joint_scales(self, spec: FamilySpec, proportions: torch.Tensor) -> torch.Tensor:
        """``(I..., J)`` accumulated uniform joint scales of the proportions ``(I..., P)``."""
        logs = torch.einsum("...p,jp->...j", proportions, self.scale_basis)
        # exp2 joint by joint (torch's scalar path) matches the float32 oracle on every golden; one vectorised exp2
        # over all joints differs by up to an ulp, more than the 1e-7 scale gate allows above 1.
        scales = [torch.exp2(logs[..., 0])]
        for joint, parent in enumerate(spec.parents[1:], start=1):
            scales.append(scales[parent] * torch.exp2(logs[..., joint]))
        return torch.stack(scales, dim=-1)

    def joint_ids(self, spec: FamilySpec, model: Model) -> torch.Tensor:
        return self.native_joints

    def corrective_features(
        self, spec: FamilySpec, model: Model, *, q: torch.Tensor, identity: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """``(..., 6 (J - 2))``: 6D features of the Euler angles of every joint after ``root``."""
        entries = full_configuration(model, q).index_select(-1, self.corrective_entries)
        return pose_feature_6d_from_euler((entries @ self.corrective_rows).unflatten(-1, (-1, 3)))

    def to(self, dtype: torch.dtype, device: torch.device) -> MhrRig:
        """Rig tensors in ``dtype`` on ``device``; the BR model only moves (``build_model`` casts it)."""

        def cast(value: Model | torch.Tensor) -> Model | torch.Tensor:
            if isinstance(value, Model) or not value.is_floating_point():
                return value.to(device=device)
            return value.to(device=device, dtype=dtype)

        return MhrRig(**{field.name: cast(getattr(self, field.name)) for field in fields(self)})


def _joint_tree(
    arrays: Mapping[str, Any], offsets: torch.Tensor, pre_rotations: torch.Tensor, dtype: torch.dtype
) -> dict[str, Any]:
    """The BR tree and its :class:`MhrRig` bindings, from the rotation entries of the parameter transform.

    Keys: ``model``, ``native_joints``, ``origin_joints``, ``pose_columns``, ``host_weights``, ``corrective_entries``
    and ``corrective_rows``.
    """
    names = [str(name) for name in arrays["joint_names"]]
    parameters = [str(name) for name in arrays["model_parameter_names"]]
    parents = arrays["joint_parents"].tolist()
    lower, upper = arrays["model_parameter_lower_limits"].tolist(), arrays["model_parameter_upper_limits"].tolist()
    sparse = zip(*(arrays[f"parameter_transform_{key}"].tolist() for key in ("rows", "columns", "values")))
    # Rotation entries (entry, joint, channel, column, weight) of joints 2.., and every pose parameter's host entry.
    entries = [
        (entry, *divmod(row, CHANNELS), column, weight)
        for entry, (row, column, weight) in enumerate(sparse)
        if row // CHANNELS >= 2 and row % CHANNELS in (3, 4, 5)
    ]
    hosts: dict[int, tuple[int, float]] = {}
    for entry, _, _, column, weight in sorted(entries, key=lambda item: (item[4] != 1.0, item[0])):
        hosts.setdefault(column, (entry, weight))

    builder = ModelBuilder("mhr", gravity=gravity(str(arrays["up_axis"])))
    root = builder.add_body(names[1])
    builder.add_free_flyer_root("root", child=root, origin=torch.cat((offsets[1], pre_rotations[1])))
    bodies, first, last, entry_joints = ["world", root], ["root"], ["universe", "root"], {}
    axes = {3: (1.0, 0.0, 0.0), 4: (0.0, 1.0, 0.0), 5: (0.0, 0.0, 1.0)}
    for joint in range(2, len(names)):
        parent, origin = bodies[parents[joint]], torch.cat((offsets[joint], pre_rotations[joint]))
        chain = sorted((item for item in entries if item[1] == joint), key=lambda item: (-item[2], item[0]))
        if not chain:
            child = builder.add_body(names[joint])
            name = builder.add_fixed(f"mhr_fixed_{names[joint]}", parent=parent, child=child, origin=origin)
            parent = child
        for link, (entry, _, channel, column, weight) in enumerate(chain):
            child = builder.add_body(names[joint] if link == len(chain) - 1 else f"mhr_node_{joint}_{link}")
            host, host_weight = hosts[column]
            if entry == host:
                name = parameters[column]
                low, high = sorted((lower[column] * host_weight, upper[column] * host_weight))
                drive = {"lower": low, "upper": high}
            else:
                name = f"mhr_mimic_{entry}_{parameters[column]}"
                drive = {"mimic_source": parameters[column], "mimic_multiplier": weight / host_weight}
            builder.add_revolute(
                name, parent=parent, child=child, axis=axes[channel], origin=origin if link == 0 else None, **drive
            )
            entry_joints[entry] = name
            parent = child
        bodies.append(parent)
        first.append(entry_joints[chain[0][0]] if chain else name)
        last.append(name)

    # float64, whatever ``dtype``: limits, mimic multipliers and gravity stay exact for a float64 Body of a
    # float32 spec, and ``build_model`` rounds them to float32 exactly as a float32 build would.
    model = build_model(builder.finalize(), dtype=torch.float64)
    dims = (model.nq, model.nv, model.nq_full, model.nv_full)
    if dims != (132, 131, 179, 178) or model.joint_names[1] != "root" or model.idx_qs[1] != 0:
        raise ValueError(f"unexpected MHR tree: (nq, nv, nq_full, nv_full) = {dims}, root free flyer not at q[:7]")
    # Every host angle has one q slot after the root: sorting the hosts by slot gives the q order.
    pose_columns = sorted(hosts, key=lambda column: model.idx_qs[model.joint_id(parameters[column])])
    full_slots = [model.idx_qs_full[model.joint_id(entry_joints[entry])] for entry, *_ in entries]
    euler_ids = torch.tensor([(joint - 2) * 3 + channel - 3 for _, joint, channel, _, _ in entries])
    return {
        "model": model,
        "native_joints": torch.tensor([model.joint_id(name) for name in last]),
        "origin_joints": torch.tensor([model.joint_id(name) for name in first]),
        "pose_columns": torch.tensor(pose_columns),
        "host_weights": torch.tensor([hosts[column][1] for column in pose_columns], dtype=dtype),
        "corrective_entries": torch.tensor(full_slots),
        "corrective_rows": torch.nn.functional.one_hot(euler_ids, 3 * (len(names) - 2)).to(dtype),
    }


__all__ = ["MhrRig"]
