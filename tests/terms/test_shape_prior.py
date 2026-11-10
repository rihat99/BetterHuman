"""``terms.shape_prior``: its residual is the identity group's coefficients, an L2 pull towards the mean shape."""

from __future__ import annotations

import torch

from better_human import terms


def test_shape_prior_residual_is_the_group_coefficients() -> None:
    betas = torch.tensor([[0.5, -1.0, 2.0], [0.0, 3.0, -0.25]], dtype=torch.float64)
    term = terms.shape_prior()
    assert term.reads == ("betas",)
    assert torch.equal(term.fn(betas), betas)
    assert terms.shape_prior("shape").reads == ("shape",)
