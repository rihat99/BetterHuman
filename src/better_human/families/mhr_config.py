"""Static conventions for the Momentum Human Rig body."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MHRConfig:
    """Authored conventions shared by every converted MHR LOD."""

    unit: str
    up_axis: str
    forward_axis: str
    native_pose_format: str
    lods: tuple[int, ...]
    default_lod: int
    asset_pattern: str
    zero_weight_group: str
    density_table: str | None


MHR_CONFIG = MHRConfig(
    unit="m",
    up_axis="+y",
    forward_axis="+z",
    native_pose_format="euler_xyz",
    lods=tuple(range(7)),
    default_lod=1,
    asset_pattern="converted/mhr_lod{lod}.npz",
    zero_weight_group="zero_weight",
    density_table=None,
)


__all__ = ["MHR_CONFIG", "MHRConfig"]
