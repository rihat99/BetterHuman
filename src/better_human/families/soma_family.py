"""NVIDIA SOMA family (SOMA-X v0.3.4, mid LOD): loader, classic parameter container and converters.

``soma(...)`` returns a :class:`~better_human.core.spec.FamilySpec` with a :class:`SOMARig`. Its identity group
``shape`` takes the 128 upstream ``identity_coeffs``; upstream poses enter through :func:`soma_from_classic`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, final

import torch
from better_robot.exceptions import ArgumentError, DtypeMismatchError, ShapeError
from better_robot.lie import so3

from better_human.core.classic import broadcast_configuration, classic_tensor
from better_human.core.spec import FamilySpec, MLPCorrective, Parts
from better_human.io import FormatError, weight_path
from better_human.io.load import load_model_npz
from better_human.io.mass_data import shipped_density_tables
from better_human.io.schema import SOMA_CORRECTIVE_FIELDS, SOMA_SCHEMA

from .densities import density_categories
from .soma_rig import SOMARig

if TYPE_CHECKING:
    from better_human.core.human_model import HumanModel, Identity

SHAPE_COUNT = 128
POSE_JOINT_COUNT = 77  #: public joints after the virtual ``Root``; ``Hips`` first.


def soma(
    path: str | Path | None = None,
    *,
    correctives: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> FamilySpec:
    """Load ``$BETTERHUMAN_MODELS_DIR/SOMA/soma_mid.npz`` (``tools/convert/soma``): 18056 vertices, 110 skin joints.

    The identity basis is ``shape (V, 3, 128)``. The 78 public joints are posable; the 32 twist helpers follow their
    limb segments. ``correctives=False`` drops the pose-corrective MLP. The spec is built once in ``dtype`` on
    ``device`` (decision record 0005).
    """
    if not isinstance(correctives, bool):
        raise ArgumentError(f"correctives must be bool, got {correctives!r}")
    if dtype not in (torch.float32, torch.float64):
        raise DtypeMismatchError(
            f"dtype={dtype!r} is not torch.float32 or torch.float64, the floating dtypes BetterRobot computes in; "
            "pass dtype=torch.float32 or dtype=torch.float64"
        )
    device = torch.device("cpu" if device is None else device)
    arrays = load_model_npz(
        weight_path("SOMA", "soma_mid.npz", path=path),
        SOMA_SCHEMA,
        exclude_fields=() if correctives else SOMA_CORRECTIVE_FIELDS,
    )

    def floats(name: str) -> torch.Tensor:
        return arrays[name].to(device=device, dtype=dtype)

    corrective = None
    if correctives:
        if bool(arrays["corrective_tanh"]):
            raise FormatError(
                "SOMA correctives with a tanh hidden layer are not supported; load with correctives=False"
            )
        rows, columns = arrays["corrective_output_indices"].to(device)
        values = floats("corrective_output_values")
        output = values.new_zeros(arrays["corrective_output_shape"].tolist()).index_put_((rows, columns), values)
        corrective = MLPCorrective(floats("corrective_first_layer").T.contiguous(), output)
    public_ids = arrays["public_joint_ids"].to(device)
    strongest = public_ids[arrays["public_skin_weights"].to(device).argmax(-1)]  # twist helpers count for their parent
    part_joints, part_ids = torch.unique(strongest, return_inverse=True)
    joint_names = tuple(str(name) for name in arrays["joint_names"])
    density_tables = shipped_density_tables()
    categories = density_categories([joint_names[joint] for joint in part_joints.tolist()], density_tables)
    return FamilySpec(
        name="soma",
        template_vertices=floats("shape_mean"),
        faces=arrays["faces"].to(device),
        joint_names=joint_names,
        parents=tuple(arrays["joint_parents"].tolist()),
        file_joint_ids=torch.arange(len(joint_names), device=device),
        skin_weights=floats("skin_weights"),
        skin_indices=arrays["skin_indices"].to(device),
        identity_bases={"shape": floats("shape_basis")},
        stream_bases={},
        corrective=corrective,
        rig=SOMARig.from_arrays(arrays, dtype=dtype, device=device),
        parts=Parts(part_ids, tuple(part_joints.tolist()), categories, density_tables),
        up_axis=str(arrays["up_axis"]),
        provenance={"archive": str(arrays["provenance"])},
    )


@final
@dataclass(frozen=True)
class SOMAClassic:
    """Upstream ``SOMALayer`` inputs.

    ``identity_coeffs (..., 128)``; ``poses (..., 77, 3)`` axis-angle per public joint relative to its T-pose
    orientation, ``Hips`` first; ``transl (..., 3)`` world position of the ``Hips`` joint in metres.
    """

    identity_coeffs: torch.Tensor
    poses: torch.Tensor
    transl: torch.Tensor


def soma_from_classic(
    human: HumanModel, classic: SOMAClassic
) -> tuple[Identity, torch.Tensor, dict[str, torch.Tensor]]:
    """``(Identity, q (..., 311), {})`` for ``human`` (a :class:`~better_human.core.human_model.HumanModel` of ``soma``).

    The identity bakes over ``identity_coeffs``' own batch and ``q`` broadcasts with it. ``q`` holds the world-aligned
    Hips free flyer ``(transl, exp(a_Hips))`` then ``exp(orient_j^T a_j)`` per public joint, ``orient_j`` its T-pose
    world rotation, so ``q_neutral`` is the T-pose.
    """
    exemplar = human.spec.template_vertices
    identity = classic_tensor("identity_coeffs", classic.identity_coeffs, SHAPE_COUNT, exemplar)
    poses = classic_tensor("poses", classic.poses, 3, exemplar)
    transl = classic_tensor("transl", classic.transl, 3, exemplar)
    if identity is None or poses is None or transl is None:
        raise TypeError("identity_coeffs, poses and transl must be floating torch.Tensors")
    if poses.shape[-2:] != (POSE_JOINT_COUNT, 3):
        raise ShapeError(f"poses must have shape (..., {POSE_JOINT_COUNT}, 3); got {tuple(poses.shape)}")
    orient = human.spec.rig.t_pose_rotations[2:]
    rotations = so3.exp(torch.cat((poses[..., :1, :], (orient.mT @ poses[..., 1:, :, None]).squeeze(-1)), dim=-2))
    batch = torch.broadcast_shapes(transl.shape[:-1], rotations.shape[:-2])
    q = torch.cat((transl.expand(*batch, 3), rotations.expand(*batch, POSE_JOINT_COUNT, 4).flatten(-2)), dim=-1)
    return human.bake(shape=identity), q, {}


def soma_to_classic(identity: Identity, q: torch.Tensor) -> SOMAClassic:
    """Classic parameters for a baked SOMA ``identity`` and ``q (..., 311)``; angles on the principal branch."""
    shape = identity.values["shape"]
    q, batch = broadcast_configuration(q, identity.model, shape)
    orient = identity.human.spec.rig.t_pose_rotations[2:]
    rotations = so3.log(so3.normalize(q[..., 3:].unflatten(-1, (POSE_JOINT_COUNT, 4))))
    poses = torch.cat((rotations[..., :1, :], (orient @ rotations[..., 1:, :, None]).squeeze(-1)), dim=-2)
    return SOMAClassic(shape.expand(*batch, SHAPE_COUNT), poses, q[..., :3].clone())


__all__ = ["SOMAClassic", "soma", "soma_from_classic", "soma_to_classic"]
