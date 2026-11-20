"""Where the parity goldens computed from licensed weights live: the models directory, never the repository.

They sit in ``$BETTERHUMAN_MODELS_DIR/goldens/<family>/`` and run in the release tier only. Call
``golden_dir`` inside a pytest fixture, never at import time or in a parametrization, so that collecting the fast
and full tiers never needs the models directory.
"""

from __future__ import annotations

from pathlib import Path

from better_human.io import models_dir


def golden_dir(family: str) -> Path:
    """``$BETTERHUMAN_MODELS_DIR/goldens/<family>``, for example ``smpl_neutral``, ``mhr_lod1`` or ``soma_mid``."""
    return models_dir() / "goldens" / family
