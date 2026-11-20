"""Where SMPL-X's vertices lie on the MHR mesh, from MHR's authors (``assets/NOTICE`` names the source)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_MAPPING = Path(__file__).parent / "assets" / "mhr2smplx_mapping.npz"


def mhr_to_smplx_mapping(
    *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(triangle_ids (10475,) long, barycentric (10475, 3))``: the MHR triangle each SMPL-X vertex lies on.

    The triangles are those of MHR at level of detail 1 (``bh.load("mhr").spec.faces``), so SMPL-X's vertices on an
    MHR mesh ``vertices (..., V, 3)`` are ``(vertices[..., faces[triangle_ids], :] * barycentric[..., None]).sum(-2)``.
    The table is MHR's ``mhr2smplx_mapping.npz`` (Apache-2.0), shipped unchanged; ``barycentric`` is cast to ``dtype``.
    """
    with np.load(_MAPPING, allow_pickle=False) as mapping:
        triangle_ids = torch.as_tensor(mapping["triangle_ids"], dtype=torch.long, device=device)
        barycentric = torch.as_tensor(mapping["baryc_coords"], dtype=dtype, device=device)
    return triangle_ids, barycentric
