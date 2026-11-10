"""Closed part meshes and shape-dependent part inertias (``better_human.mass``)."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest
import torch
from better_robot.lie import se3, so3
from better_robot.spatial import inertia_of_mesh

import better_human as bh
from better_robot.exceptions import ArgumentError, ShapeError

from better_human.io import weight_path
from better_human.io.mass_data import shipped_density_tables, shipped_segmentation
from better_human.mass import PartTopology, part_inertias, segment_vertices
from tests.support.marks import FLOAT32_GRADCHECK

CONSERVATION_RTOL = 2e-2
AUTHORED_VOLUME_RTOL = 1e-4
COM_ATOL = 3e-3
GRADCHECK_BETAS = 10


def _parts(identity: bh.Identity) -> torch.Tensor:
    """Part inertias ``(..., P, 10)`` in part order, read from the baked model's bodies."""
    return identity.model.body_inertias[..., identity.human.part_body_ids, :]


@pytest.fixture(scope="module")
def mass_body() -> bh.HumanModel:
    """The neutral SMPL model at a uniform 1000 kg/m³, the density the volume checks below divide by."""
    return bh.HumanModel.from_spec(bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz")), density=1000.0)


def _box_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=torch.long,
    )
    return vertices, faces


def _assert_closed_edges(faces: torch.Tensor) -> None:
    edges = (
        torch.cat(
            (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]),
            dim=0,
        )
        .sort(dim=-1)
        .values
    )
    _, counts = torch.unique(edges, dim=0, return_counts=True)
    torch.testing.assert_close(counts, torch.full_like(counts, 2))


def test_segment_vertices_selects_strongest_padded_influence() -> None:
    weights = torch.tensor([[0.2, 0.8, 0.0], [0.6, 0.1, 0.3]], dtype=torch.float32)
    indices = torch.tensor([[0, 3, 0], [2, 1, 4]], dtype=torch.long)
    torch.testing.assert_close(segment_vertices(weights, indices), torch.tensor([3, 2]))


def test_part_meshes_caps_mixed_face_partition_as_closed_surfaces() -> None:
    vertices, faces = _box_mesh()
    part_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    num_parts = int(part_ids.max().item()) + 1
    meshes = PartTopology.from_part_ids(faces, part_ids, num_parts=num_parts).meshes(vertices)
    masses = []
    for part_vertices, part_faces in meshes:
        _assert_closed_edges(part_faces)
        masses.append(inertia_of_mesh(part_vertices, part_faces)[0])
    torch.testing.assert_close(torch.stack(masses).sum(), inertia_of_mesh(vertices, faces)[0])


def test_authored_caps_accept_rotated_se3_part_frames() -> None:
    vertices = torch.tensor(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=torch.float32,
    )
    faces = torch.tensor(((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)), dtype=torch.long)
    topology = PartTopology(
        vertex_ids=(torch.arange(3),),
        cap_joint_ids=(torch.tensor([0]),),
        cap_centroid_ids=((),),
        faces=(faces,),
    )
    pose = torch.cat(
        (
            torch.tensor((0.0, 0.0, 1.0)),
            so3.from_euler(torch.tensor((0.2, -0.3, 0.4))),
        )
    ).unsqueeze(0)

    actual = part_inertias(vertices, pose, topology=topology, density=torch.tensor(1000.0))[0]
    capped = torch.cat((vertices, pose[:, :3]), dim=0)
    local = se3.act(se3.inverse(pose)[..., None, :], capped).squeeze(0)
    expected = inertia_of_mesh(local, faces, density=1000.0)
    torch.testing.assert_close(actual, expected)


@pytest.mark.licensed
def test_uniform_density_opt_out_and_per_part_density(mass_body: bh.HumanModel) -> None:
    assert mass_body.part_topology is not None
    assert mass_body.part_topology.num_parts == len(mass_body.spec.joint_names)
    torch.testing.assert_close(mass_body.density, torch.full_like(mass_body.density, 1000.0))

    opt_out = bh.HumanModel.from_spec(bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz")), density=None)
    assert opt_out.part_topology is None

    densities = torch.linspace(900.0, 1100.0, len(mass_body.spec.joint_names))
    varied = part_inertias(
        mass_body.bake().rest_vertices,
        mass_body.bake().rest_joint_translations,
        topology=mass_body.part_topology,
        density=densities,
    )
    expected_ratio = densities / mass_body.density
    torch.testing.assert_close(
        varied[:, 0] / _parts(mass_body.bake())[:, 0],
        expected_ratio,
        rtol=2e-5,
        atol=2e-5,
    )

    with pytest.raises(ShapeError, match=r"density must be a \(\) or \(24,\) tensor; got \(23,\)"):
        bh.HumanModel.from_spec(bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz")), density=torch.ones(23))


@pytest.mark.licensed
def test_a_bad_density_or_degenerate_part_raises_at_build() -> None:
    """Bakes integrate the parts unchecked (no host sync); the build checks density and part meshes once."""
    spec = bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz"))
    for density in (-1.0, float("nan")):
        with pytest.raises(ArgumentError, match="density must contain only finite positive values"):
            bh.HumanModel.from_spec(spec, density=density)
    flat = spec.template_vertices.clone()
    flat[..., 2] = 0.0
    with pytest.raises(ArgumentError, match="degenerate enclosed volume"):
        bh.HumanModel.from_spec(replace(spec, template_vertices=flat), density=1000.0)


@pytest.mark.licensed
@pytest.mark.parametrize(
    ("use_hands", "use_face", "expected_parts"),
    [(True, True, 55), (False, True, 25), (True, False, 52), (False, False, 22)],
)
def test_smplx_full_and_partial_spawns_receive_mass(use_hands: bool, use_face: bool, expected_parts: int) -> None:
    """At the default density (``"Dempster"``), every part of every SMPL-X spawn gets a positive mass."""
    body = bh.HumanModel.from_spec(
        bh.families.smplx(weight_path("smplx", "SMPLX_NEUTRAL.npz"), hands=use_hands, face=use_face)
    )
    parts = _parts(body.bake())
    assert parts.shape == (expected_parts, 10)
    assert torch.isfinite(parts).all()
    assert parts[:, 0].min() > 0


@pytest.mark.licensed
def test_neutral_mass_conservation_and_com_plausibility(mass_body: bh.HumanModel) -> None:
    packed = _parts(mass_body.bake())
    part_mass = packed[:, 0].sum()
    whole = inertia_of_mesh(
        mass_body.bake().rest_vertices,
        mass_body.spec.faces,
        density=1000.0,
    )
    relative_error = (part_mass - whole[0]).abs() / whole[0]
    assert relative_error <= CONSERVATION_RTOL
    assert 70.0 <= part_mass <= 90.0

    global_part_com = packed[:, 1:4] + mass_body.bake().rest_joint_translations
    aggregate_com = (packed[:, 0, None] * global_part_com).sum(dim=0) / part_mass
    torch.testing.assert_close(aggregate_com, whole[1:4], rtol=0, atol=COM_ATOL)
    lower = mass_body.bake().rest_vertices.amin(dim=-2)
    upper = mass_body.bake().rest_vertices.amax(dim=-2)
    assert bool(((aggregate_com >= lower) & (aggregate_com <= upper)).all())
    assert aggregate_com.norm() < 0.1


@pytest.mark.licensed
def test_argmax_segmentation_cross_check_against_canonical_membership(mass_body: bh.HumanModel) -> None:
    """The argmax parts against the mesh vertices of the authored closed parts."""
    num_vertices, closed_faces = shipped_segmentation("smpl")
    memberships = {
        part: {vertex for vertex in torch.unique(faces).tolist() if vertex < num_vertices}
        for part, faces in closed_faces.items()
    }
    mismatches = sum(
        vertex not in memberships[part] for vertex, part in enumerate(mass_body.spec.parts.ids.cpu().tolist())
    )
    overlap_counts = Counter(vertex for vertices in memberships.values() for vertex in vertices)
    assert mismatches == 113
    assert sum(count > 1 for count in overlap_counts.values()) == 580  # 582 with the two shoulder fin faces


@pytest.mark.licensed
def test_authored_segmentation_volume_matches_whole_mesh(mass_body: bh.HumanModel) -> None:
    packed = _parts(mass_body.bake())
    part_volume = (packed[:, 0] / mass_body.density).sum()
    whole = inertia_of_mesh(mass_body.bake().rest_vertices, mass_body.spec.faces, density=1000.0)
    whole_volume = whole[0] / 1000.0
    relative_error = (part_volume - whole_volume).abs() / whole_volume
    assert relative_error < AUTHORED_VOLUME_RTOL, f"authored volume gap {relative_error * 1e6:.1f} ppm"
    total_mass = packed[:, 0].sum()
    assert 74.0 <= total_mass <= 75.5


@pytest.mark.licensed
def test_authored_segmentation_is_batched_and_finite(mass_body: bh.HumanModel) -> None:
    betas = torch.stack((torch.zeros(GRADCHECK_BETAS), torch.linspace(-0.4, 0.4, GRADCHECK_BETAS)))
    inertias = _parts(mass_body.bake(betas=betas))
    assert inertias.shape == (2, len(mass_body.spec.joint_names), 10)
    assert torch.isfinite(inertias).all()
    assert inertias[..., 0].min() > 0


@pytest.mark.licensed
def test_named_density_table_feeds_per_part_density(mass_body: bh.HumanModel) -> None:
    body = bh.HumanModel.from_spec(bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz")), density="Chandler")
    chandler = body.density
    assert chandler.shape == (len(mass_body.spec.joint_names),)
    torch.testing.assert_close(chandler[0], chandler.new_tensor(853.0))
    torch.testing.assert_close(chandler[16], chandler.new_tensor(1005.0))

    part_volumes = _parts(body.bake())[:, 0] / chandler
    reference_volumes = _parts(mass_body.bake())[:, 0] / mass_body.density
    torch.testing.assert_close(part_volumes, reference_volumes, rtol=2e-5, atol=2e-5)


@pytest.mark.licensed
def test_named_density_table_maps_smplx_parts_by_name() -> None:
    table = shipped_density_tables()["Dempster"]
    smpl_names = list(bh.families.smpl(weight_path("smpl", "SMPL_NEUTRAL.npz")).joint_names)
    spec = bh.families.smplx(weight_path("smplx", "SMPLX_NEUTRAL.npz"), face=False)
    uniform = bh.HumanModel.from_spec(spec, density=1000.0)
    body = bh.HumanModel.from_spec(spec)
    names = list(body.spec.joint_names)
    density = body.density
    assert density.shape == (len(names),)
    for index, name in enumerate(names):
        # SMPL's own joints keep their entry; a finger takes its hand's entry.
        category = name if name in smpl_names else f"{name.split('_')[0]}_hand"
        assert float(density[index]) == pytest.approx(table[category])

    # The same part volumes, so every packed inertia row scales with its own density —
    # at the neutral shape and after shaping.
    betas = torch.linspace(-0.4, 0.4, 16)
    for dempster, plain in (
        (_parts(body.bake()), _parts(uniform.bake())),
        (_parts(body.bake(betas=betas)), _parts(uniform.bake(betas=betas))),
    ):
        torch.testing.assert_close(dempster / density[:, None], plain / uniform.density, rtol=2e-5, atol=2e-5)


@pytest.mark.licensed
def test_argmax_fallback_integrates_smplx_parts() -> None:
    body = bh.HumanModel.from_spec(bh.families.smplx(weight_path("smplx", "SMPLX_NEUTRAL.npz")))
    topology = body.part_topology
    assert topology is not None
    for vertices, faces in topology.meshes(body.bake().rest_vertices):
        assert inertia_of_mesh(vertices, faces, density=1000.0)[0] > 0


@FLOAT32_GRADCHECK
@pytest.mark.licensed
def test_float32_gradcheck_betas_to_total_mass(mass_body: bh.HumanModel) -> None:
    generator = torch.Generator().manual_seed(0)
    betas = (torch.randn(GRADCHECK_BETAS, dtype=torch.float32, generator=generator) * 0.03).requires_grad_()

    def total_mass(value: torch.Tensor) -> torch.Tensor:
        return _parts(mass_body.bake(betas=value))[:, 0].sum()

    assert torch.autograd.gradcheck(
        total_mass,
        (betas,),
        eps=1e-3,
        atol=4e-2,
        rtol=5e-2,
        fast_mode=True,
    )
