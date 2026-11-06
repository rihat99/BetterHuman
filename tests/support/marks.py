"""pytest marks that several test modules share."""

from __future__ import annotations

import pytest

# float32 is the runtime dtype these gradchecks test on purpose, with a float32 step and tolerances; torch warns that
# gradcheck expects float64.
FLOAT32_GRADCHECK = pytest.mark.filterwarnings(
    "ignore:Input #\\d+ requires gradient and is not a double precision:UserWarning"
)
