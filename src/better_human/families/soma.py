"""NVIDIA SOMA family (SOMA-X v0.3.4, mid LOD): loader, classic parameter container and converters.

``soma(...)`` returns a :class:`~better_human.core.spec.FamilySpec` with a :class:`SomaRig`. Its identity group
``shape`` takes the 128 upstream ``identity_coeffs``; upstream poses enter through :func:`soma_from_classic`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from better_robot.lie import so3

from better_human.core.classic import broadcast_configuration, classic_tensor
from better_human.core.spec import Corrective, FamilySpec, Parts
from better_human.io import SOMA_CORRECTIVE_FIELDS, SOMA_SCHEMA, load_model_npz, resolve_model_path

from .soma_rig import SomaRig

if TYPE_CHECKING:
    from better_human.core.body import Body, Identity

SHAPE_COUNT = 128
POSE_JOINT_COUNT = 77  #: public joints after the virtual ``Root``; ``Hips`` first.
METADATA = {"format_version": 1, "lod": "mid", "unit": "m", "up_axis": "+y", "forward_axis": "+z"}


def soma(
    path: str | Path | None = None,
    *,
    correctives: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> FamilySpec:
    """Load ``SOMA/soma_mid.npz`` (``tools/convert/soma``): 18056 vertices, 110 skin joints, ``shape (V, 3, 128)``.

    The 78 public joints are posable; the 32 twist helpers follow their limb segments. ``correctives=False`` drops
    the pose-corrective MLP. ``dtype=torch.float64`` here (or in ``load``) derives the rig constants in float64;
    ``Body(spec, dtype=torch.float64)`` upcasts a float32 spec's rounded ones (classic round trip 6e-8, not 1e-15).
    """
    if not isinstance(correctives, bool):
        raise TypeError("correctives must be bool")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError(f"dtype must be torch.float32 or torch.float64, got {dtype}")
    device = torch.device("cpu" if device is None else device)
    arrays = load_model_npz(
        resolve_model_path("SOMA", "soma_mid.npz", path),
        SOMA_SCHEMA,
        exclude_fields=() if correctives else SOMA_CORRECTIVE_FIELDS,
    )
    metadata = {name: arrays[name].item() for name in METADATA}
    if metadata != METADATA:
        raise ValueError(f"unsupported SOMA archive {metadata}; expected {METADATA}")

    def floats(name: str) -> torch.Tensor:
        return arrays[name].to(device=device, dtype=dtype)

    corrective = Corrective("none")
    if correctives:
        if bool(arrays["corrective_tanh"]):
            raise ValueError("SOMA correctives with a tanh hidden layer are not supported")
        rows, columns = arrays["corrective_output_indices"].to(device)
        values = floats("corrective_output_values")
        output = values.new_zeros(arrays["corrective_output_shape"].tolist()).index_put_((rows, columns), values)
        corrective = Corrective("mlp", first_layer=floats("corrective_first_layer").T.contiguous(), output=output)
    public_ids = arrays["public_joint_ids"].to(device)
    strongest = public_ids[arrays["public_skin_weights"].to(device).argmax(-1)]  # twist helpers count for their parent
    part_joints, part_ids = torch.unique(strongest, return_inverse=True)
    return FamilySpec(
        name="soma",
        vertices=floats("shape_mean"),
        faces=arrays["faces"].to(device),
        joint_names=tuple(str(name) for name in arrays["joint_names"]),
        parents=tuple(arrays["joint_parents"].tolist()),
        skin_weights=floats("skin_weights"),
        skin_indices=arrays["skin_indices"].to(device),
        identity={"shape": floats("shape_basis")},
        streams={},
        corrective=corrective,
        rig=SomaRig.from_arrays(arrays, dtype=dtype, device=device),
        parts=Parts(part_ids, tuple(part_joints.tolist())),
        unit=METADATA["unit"],
        up_axis=METADATA["up_axis"],
        forward_axis=METADATA["forward_axis"],
    )


@dataclass(frozen=True)
class SOMAClassic:
    """Upstream ``SOMALayer`` inputs.

    ``identity_coeffs (..., 128)``; ``poses (..., 77, 3)`` axis-angle per public joint relative to its T-pose
    orientation, ``Hips`` first; ``transl (..., 3)`` world position of the ``Hips`` joint in metres.
    """

    identity_coeffs: torch.Tensor
    poses: torch.Tensor
    transl: torch.Tensor


def soma_from_classic(body: Body, classic: SOMAClassic) -> tuple[Identity, torch.Tensor, dict[str, torch.Tensor]]:
    """``(Identity, q (..., 311), {})`` for ``body`` (a :class:`~better_human.core.body.Body` of ``soma``).

    The identity bakes over ``identity_coeffs``' own batch and ``q`` broadcasts with it. ``q`` holds the world-aligned
    Hips free flyer ``(transl, exp(a_Hips))`` then ``exp(orient_j^T a_j)`` per public joint, ``orient_j`` its T-pose
    world rotation, so ``q_neutral`` is the T-pose.
    """
    exemplar = body.spec.vertices
    identity = classic_tensor("identity_coeffs", classic.identity_coeffs, SHAPE_COUNT, exemplar)
    poses = classic_tensor("poses", classic.poses, 3, exemplar)
    transl = classic_tensor("transl", classic.transl, 3, exemplar)
    if identity is None or poses is None or transl is None:
        raise TypeError("identity_coeffs, poses and transl must be floating torch.Tensors")
    if poses.shape[-2:] != (POSE_JOINT_COUNT, 3):
        raise ValueError(f"poses must have shape (..., {POSE_JOINT_COUNT}, 3); got {tuple(poses.shape)}")
    orient = body.spec.rig.t_pose_rotations[2:]
    rotations = so3.exp(torch.cat((poses[..., :1, :], (orient.mT @ poses[..., 1:, :, None]).squeeze(-1)), dim=-2))
    batch = torch.broadcast_shapes(transl.shape[:-1], rotations.shape[:-2])
    q = torch.cat((transl.expand(*batch, 3), rotations.expand(*batch, POSE_JOINT_COUNT, 4).flatten(-2)), dim=-1)
    return body.bake(shape=identity), q, {}


def soma_to_classic(identity: Identity, q: torch.Tensor) -> SOMAClassic:
    """Classic parameters for a baked SOMA ``identity`` and ``q (..., 311)``; angles on the principal branch."""
    shape = identity.params["shape"]
    q, batch = broadcast_configuration(q, identity.model, shape)
    orient = identity.spec.rig.t_pose_rotations[2:]
    rotations = so3.log(so3.normalize(q[..., 3:].unflatten(-1, (POSE_JOINT_COUNT, 4))))
    poses = torch.cat((rotations[..., :1, :], (orient @ rotations[..., 1:, :, None]).squeeze(-1)), dim=-2)
    return SOMAClassic(shape.expand(*batch, SHAPE_COUNT), poses, q[..., :3].clone())


__all__ = ["SOMAClassic", "SomaRig", "soma", "soma_from_classic", "soma_to_classic"]
