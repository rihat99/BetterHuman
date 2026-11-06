"""``load_model_npz`` returns every schema field of a licensed archive exactly as stored.

The oracle is ``np.load`` of the same archive; equal means equal dtype, shape and bytes. The loader promises never to
cast, and a faster reader (decision record 0013) must keep that promise.
"""

from __future__ import annotations

import numpy as np
import pytest

from better_human.io import MHR_SCHEMA, SMPL_SCHEMA, SMPLX_SCHEMA, SOMA_SCHEMA, load_model_npz, weight_path

pytestmark = pytest.mark.licensed

ARCHIVES = {
    "smpl": ("smpl", "SMPL_NEUTRAL.npz", SMPL_SCHEMA),
    "smplx": ("smplx", "SMPLX_NEUTRAL.npz", SMPLX_SCHEMA),
    "mhr_lod1": ("MHR", "converted/mhr_lod1.npz", MHR_SCHEMA),
    "soma": ("SOMA", "soma_mid.npz", SOMA_SCHEMA),
}


@pytest.mark.parametrize("family", ARCHIVES)
def test_reader_is_bitwise_np_load(family: str) -> None:
    folder, filename, schema = ARCHIVES[family]
    path = weight_path(folder, filename)
    loaded = load_model_npz(path, schema)
    assert set(loaded) == set(schema.fields)
    with np.load(path, allow_pickle=False) as archive:
        for name, value in loaded.items():
            actual = value if isinstance(value, np.ndarray) else value.numpy()
            expected = archive[name]
            assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape), name
            assert actual.tobytes() == expected.tobytes(), name
