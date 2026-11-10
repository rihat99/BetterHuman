"""NVIDIA SOMA family (SOMA-X v0.3.4, mid LOD) loader.

``soma(...)`` returns a :class:`~better_human.core.spec.FamilySpec` with a :class:`SOMARig`. Its identity group
``shape`` takes the 128 upstream ``identity_coeffs``; upstream poses enter through ``better_human.from_classic`` with a
:class:`~better_human.families.SOMAClassic`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from better_robot.exceptions import DtypeMismatchError

from better_human.core.spec import FamilySpec, MLPCorrective, Parts
from better_human.io import FormatError, weight_path
from better_human.io.load import load_model_npz
from better_human.io.mass_data import shipped_density_tables
from better_human.io.schema import SOMA_CORRECTIVE_FIELDS, SOMA_SCHEMA

from ._checks import check_flags
from .classic import SOMAFormat
from .soma_rig import SOMARig

#: The density category of each part, by its joint (decision record 0007). The categories are SMPL's part names, each
#: named after the joint at its proximal end, so a part takes the SMPL part that covers the same body segment: SOMA's
#: Shoulder is the clavicle (SMPL collar), Arm the upper arm (SMPL shoulder), Leg the thigh (SMPL hip), Hand the palm
#: (SMPL wrist), Foot the foot (SMPL ankle) and ToeBase the toes (SMPL foot).
_DENSITY_CATEGORIES = {
    "Hips": "pelvis",
    "Spine1": "spine1",
    "Spine2": "spine2",
    "Chest": "spine3",
    "Neck1": "neck",
    "Neck2": "neck",
    "Head": "head",
    "Jaw": "head",
    **{
        f"{side.title()}{segment}": f"{side}_{category}"
        for side in ("left", "right")
        for segment, category in (
            ("Shoulder", "collar"),
            ("Arm", "shoulder"),
            ("ForeArm", "elbow"),
            ("Hand", "wrist"),
            ("Leg", "hip"),
            ("Shin", "knee"),
            ("Foot", "ankle"),
            ("ToeBase", "foot"),
        )
    },
    **{
        f"{side.title()}Hand{finger}{segment}": f"{side}_hand"
        for side in ("left", "right")
        for finger, segments in (
            ("Thumb", "123"),
            ("Index", "1234"),
            ("Middle", "1234"),
            ("Ring", "1234"),
            ("Pinky", "1234"),
        )
        for segment in (*segments, "End")
    },
}


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
    check_flags(correctives=correctives)
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
    categories = tuple(_DENSITY_CATEGORIES.get(joint_names[joint]) for joint in part_joints.tolist())
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
        classic=SOMAFormat(),
        up_axis=str(arrays["up_axis"]),
        provenance={"archive": str(arrays["provenance"])},
    )


__all__ = ["soma"]
