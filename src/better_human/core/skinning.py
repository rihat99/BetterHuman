"""The one skinning path of every rig: blend shapes, skin affines, linear blend skinning."""

from __future__ import annotations

import torch


def blend_shapes(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Blend linear basis columns.

    Parameters
    ----------
    coefficients
        Shape ``(..., S)``.
    basis
        Shape ``(V, 3, S)``.

    Returns
    -------
    torch.Tensor
        Displacements with shape ``(..., V, 3)``.
    """
    return torch.einsum("...s,vcs->...vc", coefficients, basis)


def skin_affines(joint_transforms: torch.Tensor, inverse_bind_affines: torch.Tensor) -> torch.Tensor:
    """Per-joint rest-to-world affines ``(..., J, 3, 4)``: world ``[R | p]`` joint transforms ``(..., J, 3, 4)``
    composed with ``inverse_bind_affines (..., J, 3, 4)``, each the affine from the rest frame into its joint's frame.

    Every rig skins through this one path: the SMPL family binds ``[I | -rest_joints]``, SOMA
    its authored binds, MHR its binds with the joint scales folded in.
    """
    rotation, translation = joint_transforms[..., :3], joint_transforms[..., 3]
    affine_translation = (rotation @ inverse_bind_affines[..., 3:]).squeeze(-1) + translation
    return torch.cat((rotation @ inverse_bind_affines[..., :3], affine_translation[..., None]), dim=-1)


def linear_blend_skinning(rest_vertices: torch.Tensor, affines: torch.Tensor, *, weights: torch.Tensor) -> torch.Tensor:
    """Skinned vertices ``(..., V, 3)`` of ``rest_vertices (..., V, 3)``, ``affines (..., J, 3, 4)`` and ``weights (V, J)``."""
    batch = torch.broadcast_shapes(rest_vertices.shape[:-2], affines.shape[:-3])
    flat = affines.expand(*batch, *affines.shape[-3:]).flatten(-2)
    # The blended affine [A | t] of each vertex, as A's three columns and t, each (..., V, 3).
    columns = torch.einsum("vj,...jx->...vx", weights, flat).unflatten(-1, (3, 4)).unbind(-1)
    # A x + t as the columns of A scaled by the coordinates of x, accumulated into one new tensor: a batched 3x3
    # product per vertex, or a temporary per term, is three to four times slower at batch 64.
    x, y, z = rest_vertices.unsqueeze(-2).unbind(-1)  # each (..., V, 1)
    return (columns[0] * x).addcmul_(columns[1], y).addcmul_(columns[2], z).add_(columns[3])


def top_k_skin(weights: torch.Tensor, k: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense ``(V, J)`` weights to padded ``(V, K)`` values and indices (``K`` = largest row support)."""
    nonzero = weights != 0
    if k is None:
        k = int(nonzero.sum(dim=-1).max().item())
    order = torch.argsort(nonzero.to(torch.int8), dim=-1, descending=True, stable=True)[:, :k]
    values = torch.gather(weights, 1, order)
    indices = torch.where(values != 0, order, torch.zeros_like(order))
    return values, indices


__all__ = ["blend_shapes", "linear_blend_skinning", "skin_affines", "top_k_skin"]
