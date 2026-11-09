"""The two correctives against their reference formulas, on the full mesh and a vertex subset, with gradchecks."""

from __future__ import annotations

import torch
from torch.nn import functional

from better_human.core import LinearCorrective, MLPCorrective


def test_correctives_match_their_formulas_and_gradcheck() -> None:
    generator = torch.Generator().manual_seed(11)
    features, hidden, vertices = 5, 3, 4
    feature = torch.randn(2, 3, features, generator=generator, dtype=torch.float64)
    posedirs = torch.randn(features, vertices * 3, generator=generator, dtype=torch.float64)
    first_layer = torch.randn(hidden, features, generator=generator, dtype=torch.float64)
    output = torch.randn(hidden, vertices * 3, generator=generator, dtype=torch.float64)
    linear, mlp = LinearCorrective(posedirs), MLPCorrective(first_layer, output)
    subset = torch.tensor([3, 1])

    expected_linear = (feature @ posedirs).reshape(2, 3, vertices, 3)
    expected_mlp = (functional.relu(feature @ first_layer.T) @ output).reshape(2, 3, vertices, 3)
    original = feature.clone()
    torch.testing.assert_close(linear.apply(feature), expected_linear, rtol=0, atol=1e-15)
    torch.testing.assert_close(mlp.apply(feature), expected_mlp, rtol=0, atol=1e-15)
    torch.testing.assert_close(linear.apply(feature, subset), expected_linear[..., subset, :], rtol=0, atol=1e-15)
    torch.testing.assert_close(mlp.apply(feature, subset), expected_mlp[..., subset, :], rtol=0, atol=1e-15)
    assert torch.equal(feature, original)

    probe = feature[0].clone().requires_grad_(True)
    for corrective in (linear, mlp):
        assert torch.autograd.gradcheck(lambda value, c=corrective: c.apply(value), (probe,))
        assert torch.autograd.gradcheck(lambda value, c=corrective: c.apply(value, subset), (probe,))
    weights = (posedirs.clone().requires_grad_(True), first_layer.clone().requires_grad_(True))
    assert torch.autograd.gradcheck(lambda p: LinearCorrective(p).apply(feature[0]), (weights[0],))
    assert torch.autograd.gradcheck(lambda w: MLPCorrective(w, output).apply(feature[0]), (weights[1],))
