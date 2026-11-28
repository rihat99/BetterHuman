"""Mass properties of a segmented mesh: closed part topology, part densities and part inertias.

``mass`` reads no file and knows no family: the family loaders hand it their segmentation and density tables through
``Parts``.
"""

from .parts import part_densities, part_inertias, segment_vertices
from .topology import PartTopology

__all__ = ["PartTopology", "part_densities", "part_inertias", "segment_vertices"]
