"""Generic load-time reduction for terminal joint subtrees and parameters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def slice_parameter_basis(
    basis: torch.Tensor,
    group_columns: Mapping[str, slice],
    requested_widths: Mapping[str, int],
) -> tuple[torch.Tensor, dict[str, int]]:
    """Select leading columns from named parameter-group ranges.

    ``basis`` has shape ``(V, 3, S)``. ``group_columns`` declares each
    group's source range and ``requested_widths`` declares the ordered output
    groups. The returned basis concatenates those selections in output order.
    """
    if basis.ndim != 3 or basis.shape[-2] != 3:
        raise ValueError(f"basis must have shape (V, 3, S), got {tuple(basis.shape)}")

    selected: list[torch.Tensor] = []
    widths: dict[str, int] = {}
    for name, width in requested_widths.items():
        try:
            columns = group_columns[name]
        except KeyError as error:
            raise ValueError(f"Unknown parameter group {name!r}") from error
        if (
            not isinstance(columns.start, int)
            or isinstance(columns.start, bool)
            or not isinstance(columns.stop, int)
            or isinstance(columns.stop, bool)
            or columns.step not in {None, 1}
        ):
            raise ValueError(f"Parameter group {name!r} must use a bounded contiguous slice")
        if columns.start < 0 or columns.stop < columns.start or columns.stop > basis.shape[-1]:
            raise ValueError(
                f"Parameter group {name!r} slice [{columns.start}, {columns.stop}) "
                f"must fit basis width {basis.shape[-1]}"
            )
        available = columns.stop - columns.start
        if not isinstance(width, int) or isinstance(width, bool) or width < 0 or width > available:
            raise ValueError(f"Parameter group {name!r} width must be in [0, {available}], got {width}")
        selected.append(basis[..., columns.start : columns.start + width])
        widths[name] = width
    if not selected:
        return basis[..., :0], widths
    return torch.cat(selected, dim=-1), widths


def _validate_terminal_drop(parents: Sequence[int], dropped: set[int]) -> None:
    joint_count = len(parents)
    if 0 in dropped:
        raise ValueError("The root joint cannot be dropped")
    invalid = sorted(index for index in dropped if index < 0 or index >= joint_count)
    if invalid:
        raise ValueError(f"Dropped joint indices are out of range [0, {joint_count}): {invalid}")
    for child, parent in enumerate(parents):
        if parent in dropped and child not in dropped:
            raise ValueError(
                f"Dropped joints must form terminal subtrees; joint {parent} is dropped "
                f"but child {child} survives"
            )


def _slice_regressor_rows(regressor: torch.Tensor, survivor_indices: torch.Tensor) -> torch.Tensor:
    if regressor.layout == torch.strided:
        return regressor.index_select(0, survivor_indices)
    if regressor.layout != torch.sparse_csr:
        raise ValueError(f"joint_regressor must be dense or sparse CSR, got layout {regressor.layout}")

    coo = regressor.to_sparse_coo().coalesce()
    old_to_new = torch.full(
        (regressor.shape[0],),
        -1,
        dtype=torch.long,
        device=survivor_indices.device,
    )
    old_to_new[survivor_indices] = torch.arange(
        survivor_indices.numel(),
        device=survivor_indices.device,
    )
    new_rows = old_to_new[coo.indices()[0]]
    keep = new_rows >= 0
    indices = torch.stack((new_rows[keep], coo.indices()[1, keep]))
    reduced = torch.sparse_coo_tensor(
        indices,
        coo.values()[keep],
        size=(survivor_indices.numel(), regressor.shape[1]),
        dtype=regressor.dtype,
        device=regressor.device,
        check_invariants=True,
    )
    return reduced.coalesce().to_sparse_csr()


def reduce_joint_data(
    joint_names: Sequence[str],
    parents: Sequence[int],
    joint_regressor: torch.Tensor,
    skinning_weights: torch.Tensor,
    posedirs: torch.Tensor,
    dropped_joints: Sequence[int],
) -> tuple[
    tuple[str, ...],
    tuple[int, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
]:
    """Drop terminal subtrees and return reduced native-order joint data.

    ``joint_regressor`` has shape ``(J, V)``, ``skinning_weights`` has
    shape ``(V, J)``, and ``posedirs`` has shape ``(V, 3, (J - 1) * 9)``.
    The final tuple contains names, remapped parents, regressor, normalized
    weights, sliced correctives, and surviving original joint indices.
    """
    joint_count = len(joint_names)
    if len(parents) != joint_count:
        raise ValueError("parents and joint_names must have the same length")
    if joint_regressor.ndim != 2:
        raise ValueError(f"joint_regressor must have shape (J, V), got {tuple(joint_regressor.shape)}")
    if skinning_weights.ndim != 2:
        raise ValueError(f"skinning_weights must have shape (V, J), got {tuple(skinning_weights.shape)}")
    if posedirs.ndim != 3 or posedirs.shape[1] != 3:
        raise ValueError(f"posedirs must have shape (V, 3, (J - 1) * 9), got {tuple(posedirs.shape)}")
    if joint_regressor.shape[0] != joint_count or skinning_weights.shape[1] != joint_count:
        raise ValueError("joint regressor and skinning weights must use the declared joint count")
    vertex_count = skinning_weights.shape[0]
    if joint_regressor.shape[1] != vertex_count or posedirs.shape[0] != vertex_count:
        raise ValueError("joint regressor, skinning weights, and posedirs must use the same vertex count")
    expected_pose_width = (joint_count - 1) * 9
    if posedirs.shape[-1] != expected_pose_width:
        raise ValueError(f"posedirs must have trailing width {expected_pose_width}, got {posedirs.shape[-1]}")
    weight_sums = skinning_weights.sum(dim=-1)
    if not bool(torch.isfinite(weight_sums).all()) or bool((weight_sums <= 0).any()):
        raise ValueError("skinning weight rows must have positive finite sums")

    dropped = set(dropped_joints)
    _validate_terminal_drop(parents, dropped)
    if not dropped:
        return (
            tuple(joint_names),
            tuple(parents),
            joint_regressor,
            skinning_weights,
            posedirs,
            tuple(range(joint_count)),
        )

    survivors = tuple(index for index in range(joint_count) if index not in dropped)
    old_to_new = {old: new for new, old in enumerate(survivors)}
    reduced_names = tuple(joint_names[index] for index in survivors)
    reduced_parents = tuple(-1 if old == 0 else old_to_new[parents[old]] for old in survivors)
    survivor_tensor = torch.tensor(survivors, dtype=torch.long, device=skinning_weights.device)
    reduced_regressor = _slice_regressor_rows(joint_regressor, survivor_tensor)

    target_indices: list[int] = []
    for joint in range(joint_count):
        ancestor = joint
        while ancestor in dropped:
            ancestor = parents[ancestor]
        target_indices.append(old_to_new[ancestor])
    target_tensor = torch.tensor(target_indices, dtype=torch.long, device=skinning_weights.device)
    reduced_weights = skinning_weights.new_zeros((skinning_weights.shape[0], len(survivors)))
    reduced_weights = reduced_weights.index_add(1, target_tensor, skinning_weights)
    reduced_weights = reduced_weights / reduced_weights.sum(dim=-1, keepdim=True)

    feature_blocks = [
        torch.arange((joint - 1) * 9, joint * 9, device=posedirs.device)
        for joint in survivors
        if joint != 0
    ]
    feature_indices = (
        torch.cat(feature_blocks)
        if feature_blocks
        else torch.empty(0, dtype=torch.long, device=posedirs.device)
    )
    reduced_posedirs = posedirs.index_select(-1, feature_indices)
    return (
        reduced_names,
        reduced_parents,
        reduced_regressor,
        reduced_weights,
        reduced_posedirs,
        survivors,
    )


__all__ = ["reduce_joint_data", "slice_parameter_basis"]
