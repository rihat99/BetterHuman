"""Find weight files, read and write frame sets, check the formats BetterHuman files carry, and read shipped data.

A reader of a BetterHuman format (the package's own, ``better_human.studio``'s) migrates what it read with
:func:`upgrade`; a writer writes :func:`header` (``docs/reference/formats/``).

The archive readers and their schemas are internal (``io.load``, ``io.schema``, ``io.mass_data``).
"""

from .formats import FormatError, check_format, header, upgrade
from .frame_sets import read_frame_set, shipped_frame_set, write_frame_set
from .load import AssetNotFoundError, models_dir, weight_path
from .mhr_smplx import mhr_to_smplx_mapping

__all__ = [
    "AssetNotFoundError",
    "FormatError",
    "check_format",
    "header",
    "mhr_to_smplx_mapping",
    "models_dir",
    "read_frame_set",
    "shipped_frame_set",
    "upgrade",
    "weight_path",
    "write_frame_set",
]
