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
    output[1, 3:] = 0  # hidden row 1 moves vertex 0 only: the blocks keep the nonzeros
    linear, mlp = LinearCorrective(posedirs), MLPCorrective.from_dense(first_layer, output, rows=1)
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
    assert torch.autograd.gradcheck(
        lambda w: MLPCorrective.from_dense(w, output, rows=1).apply(feature[0]), (weights[1],)
    )


def test_mlp_blocks_equal_the_dense_output_over_long_runs_and_stray_entries() -> None:
    generator = torch.Generator().manual_seed(0)
    hidden, features, vertices, rows = 48, 7, 300, 8
    first_layer = torch.randn(hidden, features, generator=generator, dtype=torch.float64)
    output = torch.zeros(hidden, vertices, 3, dtype=torch.float64)
    for run, count in enumerate((0, 5, 64, 65, 130, 200)):  # runs of none, part of one, one and several blocks
        moved = torch.randperm(vertices - 1, generator=generator)[:count] + 1
        output[run * rows : (run + 1) * rows, moved] = torch.randn(
            rows, count, 3, generator=generator, dtype=torch.float64
        )
    output[3, 0, 1], output[17, 0, 0] = 2.5, -1.0  # one row of a run moves vertex 0, which its other rows leave alone
    output = output.flatten(1)
    mlp = MLPCorrective.from_dense(first_layer, output, rows=rows)
    feature = torch.randn(2, 3, features, generator=generator, dtype=torch.float64)
    expected = (functional.relu(feature @ first_layer.T) @ output).unflatten(-1, (vertices, 3))
    subset = torch.tensor([299, 0, 5, 5, 17])

    torch.testing.assert_close(mlp.apply(feature), expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(mlp.apply(feature, subset), expected[..., subset, :], rtol=0, atol=1e-12)
