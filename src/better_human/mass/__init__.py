"""Mass properties of a segmented mesh: closed part topology, part densities and part inertias (record 0007).

``mass`` reads no file and knows no family: the family loaders hand it their segmentation and density tables through
``Parts`` (decision record 0011).
"""

from .densities import part_densities
from .inertia import part_inertias
from .segmentation import segment_vertices
from .topology import PartTopology

__all__ = ["PartTopology", "part_densities", "part_inertias", "segment_vertices"]
