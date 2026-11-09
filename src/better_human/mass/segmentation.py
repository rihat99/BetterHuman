"""Vertex segmentation: each vertex's part from its skin weights."""

from __future__ import annotations

import torch


def segment_vertices(weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """``(V,)`` long: the joint of each vertex's strongest influence.

    :param weights: ``(V, K)`` nonnegative top-K skin weights.
    :param indices: ``(V, K)`` joint of each weight.
    """
    strongest = weights.argmax(dim=-1, keepdim=True)
    return indices.gather(-1, strongest).squeeze(-1).to(dtype=torch.long)
