"""Shared synthetic fixtures for I/O schema tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def mhr_data() -> dict[str, np.ndarray]:
    """Return a small format-v2 MHR archive mapping with every field the reader reads."""
    vertex_count = 4
    face_count = 2
    return {
        "format": np.asarray("better_human.mhr"),
        "format_version": np.asarray(2, dtype=np.int64),
        "provenance": np.asarray('{"converter": "synthetic"}'),
        "lod": np.asarray(6, dtype=np.int64),
        "joint_names": np.full(127, "joint", dtype="<U20"),
        "joint_parents": np.zeros(127, dtype=np.int64),
        "joint_pre_rotations": np.zeros((127, 4), dtype=np.float32),
        "joint_translation_offsets": np.zeros((127, 3), dtype=np.float32),
        "segmentation_group_names": np.full(20, "part", dtype="<U11"),
        "segmentation_joint_group_indices": np.zeros(127, dtype=np.int64),
        "model_parameter_names": np.full(204, "parameter", dtype="<U24"),
        "model_parameter_channel_types": np.zeros(204, dtype=np.int8),
        "parameter_transform_rows": np.zeros(368, dtype=np.int64),
        "parameter_transform_columns": np.zeros(368, dtype=np.int64),
        "parameter_transform_values": np.zeros(368, dtype=np.float32),
        "model_parameter_lower_limits": np.zeros(204, dtype=np.float32),
        "model_parameter_upper_limits": np.zeros(204, dtype=np.float32),
        "template_vertices": np.zeros((vertex_count, 3), dtype=np.float32),
        "faces": np.zeros((face_count, 3), dtype=np.int64),
        "skin_indices": np.zeros((vertex_count, 8), dtype=np.int64),
        "skin_weights": np.zeros((vertex_count, 8), dtype=np.float32),
        "identity_basis": np.zeros((vertex_count, 3, 45), dtype=np.float32),
        "expression_basis": np.zeros((vertex_count, 3, 72), dtype=np.float32),
        "up_axis": np.asarray("+y", dtype="<U2"),
        "bind_matrices": np.zeros((127, 4, 4), dtype=np.float32),
        "inverse_bind_matrices": np.zeros((127, 4, 4), dtype=np.float32),
        "corrective_first_layer_indices": np.zeros((2, 53136), dtype=np.int64),
        "corrective_first_layer_weights": np.zeros(53136, dtype=np.float32),
        "corrective_first_layer_mask": np.zeros((3000, 750), dtype=np.bool_),
        "corrective_output_weights": np.zeros((3000, vertex_count * 3), dtype=np.float32),
    }
