"""Shared parametric-body objects, math, and classic converters."""

from .body import BodyStructure, BodyValues, ParametricBody
from .contact_frames import (
    ContactFrames,
    frame_local_placements,
    parse_frame_set,
    resolve_parents,
    vertex_normals,
)

__all__ = [
    "BodyStructure",
    "BodyValues",
    "ContactFrames",
    "ParametricBody",
    "frame_local_placements",
    "parse_frame_set",
    "resolve_parents",
    "vertex_normals",
]
