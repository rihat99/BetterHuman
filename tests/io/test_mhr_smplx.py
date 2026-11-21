"""The MHR to SMPL-X mapping shipped as package data: a triangle of MHR at level of detail 1 per SMPL-X vertex."""

from __future__ import annotations

import torch

from better_human.io import mhr_to_smplx_mapping

MHR_LOD1_FACES = 36874


def test_the_mapping_gives_each_smplx_vertex_a_triangle_and_barycentric_weights() -> None:
    triangle_ids, barycentric = mhr_to_smplx_mapping(dtype=torch.float64)
    assert (triangle_ids.shape, triangle_ids.dtype) == ((10475,), torch.long)
    assert (barycentric.shape, barycentric.dtype) == ((10475, 3), torch.float64)
    assert int(triangle_ids.min()) >= 0 and int(triangle_ids.max()) < MHR_LOD1_FACES
    torch.testing.assert_close(barycentric.sum(-1), torch.ones(10475, dtype=torch.float64), rtol=0, atol=1e-12)
