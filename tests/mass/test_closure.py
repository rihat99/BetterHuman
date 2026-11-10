"""Every part of the SMPL family closes, and the part volumes sum to the mesh volume (H2 review finding H2).

The mesh volume is the mesh closed the way its parts are (SMPL-X has holes; one part caps each about its centroid).
Each tolerance bounds the measured residual, whose cause is:

- SMPL, authored parts: where spine3 meets a collar, three cap triangles per shoulder and the dropped shoulder face
  leave a 2.6 mL sliver in no part; measured -7.06e-5.
- SMPL-H, parts from part ids: at the hands two junction polygons, and five runs whose neighbour cuts the seam at
  another vertex (where a third part touches it at one vertex), do not cancel: +3.7e-4 and -3.8e-4, net -4.1e-6.
- SMPL-X: head and jaw cap the mouth, a mesh hole, with two runs and a chord; the mesh caps it with one fan:
  +2.18e-4 and -2.34e-4, net -1.7e-5.
- Without hands and face every junction polygon is shared, so the sum is exact up to rounding.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch
from better_robot.spatial import inertia_of_mesh

import better_human as bh
from better_human.mass import PartTopology

pytestmark = pytest.mark.licensed


def _volume(vertices: torch.Tensor, faces: torch.Tensor) -> float:
    return float(inertia_of_mesh(vertices, faces, density=1.0)[0])


def _is_closed(faces: torch.Tensor) -> bool:
    counts = Counter((a, b) for face in faces.tolist() for a, b in zip(face, face[1:] + face[:1]))
    return all(count == 1 and counts[(b, a)] == 1 for (a, b), count in counts.items())


@pytest.mark.parametrize(
    ("family", "options", "tolerance"),
    [
        ("smpl", {}, 1e-4),
        ("smplh", {}, 1e-5),
        ("smplh", {"hands": False}, 1e-12),
        ("smplx", {}, 3e-5),
        ("smplx", {"face": False}, 1e-12),
        ("smplx", {"hands": False, "face": False}, 1e-12),
    ],
    ids=["smpl", "smplh", "smplh-no-hands", "smplx", "smplx-no-face", "smplx-no-hands-no-face"],
)
def test_parts_are_closed_and_their_volumes_sum_to_the_mesh(family: str, options: dict, tolerance: float) -> None:
    spec = getattr(bh.families, family)(dtype=torch.float64, **options)
    human = bh.HumanModel.from_spec(spec, density=1000.0)
    identity = human.bake()
    meshes = human.part_topology.meshes(identity.rest_vertices, identity.rest_joint_translations)
    assert [part for part, (_, faces) in enumerate(meshes) if not _is_closed(faces)] == []
    whole = PartTopology.from_part_ids(spec.faces, torch.zeros(spec.num_vertices, dtype=torch.long), num_parts=1)
    ((mesh_vertices, mesh_faces),) = whole.meshes(identity.rest_vertices)
    mesh_volume = _volume(mesh_vertices, mesh_faces)
    part_volume = sum(_volume(vertices, faces) for vertices, faces in meshes)
    assert abs(part_volume - mesh_volume) / mesh_volume <= tolerance


@pytest.mark.parametrize(
    "dropped", [["left_hand", "right_hand"], ["left_elbow", "left_wrist", "left_hand"]], ids=["hands", "left-forearm"]
)
def test_drop_joints_merges_authored_parts_and_conserves_mass(dropped: list[str]) -> None:
    """The baked total mass is kept; each untouched part's closed mesh integrates bitwise as before.

    The baked inertias of untouched parts differ from the full model's in the last bits (about 4e-14 relative): the
    bake skins the rest mesh into the neutral pose with the reduced skin weights, whose sums round differently.
    """
    spec = bh.families.smpl(dtype=torch.float64)
    reduced = bh.families.drop_joints(spec, dropped)
    totals, parts = [], []
    for current in (spec, reduced):
        human = bh.HumanModel.from_spec(current, density=1000.0)
        identity = human.bake()
        totals.append(float(identity.model.body_inertias[..., 0].sum()))
        meshes = human.part_topology.meshes(identity.rest_vertices, identity.rest_joint_translations)
        names = [current.joint_names[joint] for joint in current.parts.joints]
        parts.append({name: inertia_of_mesh(*mesh, density=1000.0) for name, mesh in zip(names, meshes, strict=True)})
    assert abs(totals[1] - totals[0]) / totals[0] <= 1e-6
    full, merged = parts
    ancestors = {spec.joint_names[spec.parents[spec.joint_id(name)]] for name in dropped} - set(dropped)
    assert {name for name in merged if not torch.equal(merged[name], full[name])} <= ancestors


def test_drop_joints_keeps_the_volume_and_gives_the_merged_volume_the_survivor_density() -> None:
    """At a non-uniform density the merged part keeps its survivor's density (H3 review L2).

    The total volume is kept; the total mass changes by each dropped part's volume times its density difference to
    the survivor: SMPL's left forearm and hand (Dempster 1130 and 1160 kg/m³) join the upper arm (1070), -0.085 kg.
    Measured residuals (2026-11-10): volume 0, mass 2.4e-16 relative.
    """
    spec = bh.families.smpl(dtype=torch.float64)
    dropped = ["left_elbow", "left_wrist", "left_hand"]
    volumes, densities, masses = [], [], []
    for current in (spec, bh.families.drop_joints(spec, dropped)):
        human = bh.HumanModel.from_spec(current, density="Dempster")
        identity = human.bake()
        meshes = human.part_topology.meshes(identity.rest_vertices, identity.rest_joint_translations)
        names = [current.joint_names[joint] for joint in current.parts.joints]
        volumes.append(dict(zip(names, (_volume(*mesh) for mesh in meshes), strict=True)))
        densities.append(dict(zip(names, human.density.tolist(), strict=True)))
        masses.append(float(identity.model.body_inertias[..., 0].sum()))
    (full, reduced), (full_density, reduced_density) = volumes, densities
    assert abs(sum(reduced.values()) - sum(full.values())) <= 1e-12 * sum(full.values())
    assert reduced_density["left_shoulder"] == full_density["left_shoulder"]
    change = sum(full[name] * (full_density["left_shoulder"] - full_density[name]) for name in dropped)
    assert change < -0.08
    assert abs((masses[1] - masses[0]) - change) <= 1e-12 * masses[0]
