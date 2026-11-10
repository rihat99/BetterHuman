"""Render the segmentation images embedded in ``docs/concepts/segmentation.md``.

For SMPL, SMPL-X and MHR at zero shape, each segmentation the body has (authored, argmax) is drawn twice: a
part-coloured surface and an exploded view of the parts (``<family>_<authored|argmax>_<surface|split>.png``).
The argmax parts are the closed topology ``PartTopology.from_part_ids`` builds from the family's vertex labelling
(``FamilySpec.parts.ids``); SMPL's authored split is its closed topology from the shipped segmentation archive, and
the authored surfaces colour the studio's shipped per-vertex labellings (SMPL and SMPL-X). Rendering is headless
matplotlib (Agg) with a fixed camera, fixed colours and a seeded colour permutation, so a rerun writes the same
images. It reads the licensed weights under ``$BETTERHUMAN_MODELS_DIR``; a family whose weights are missing is
reported and skipped. Regenerate the images with::

    uv run python tools/render_segmentation.py
"""

from __future__ import annotations

import argparse
import json
from importlib import resources
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import better_human as bh  # noqa: E402
from better_human.mass import PartTopology  # noqa: E402

DOCS_IMAGES = Path(__file__).resolve().parents[1] / "docs" / "segmentation"
FAMILIES = ("smpl", "smplx", "mhr")
_LIGHT = np.array([0.3, -0.8, 0.5]) / np.linalg.norm([0.3, -0.8, 0.5])
_EXPLODE_GAIN = 0.55
_UNLABELLED = np.array([0.8, 0.8, 0.8])

Meshes = list[tuple[np.ndarray, np.ndarray]]


def _colors(num_parts: int) -> np.ndarray:
    """``(num_parts, 3)`` distinct RGB colours, shuffled by a fixed seed so neighbouring parts contrast."""
    samples = matplotlib.colormaps["gist_rainbow"](np.linspace(0.0, 1.0, num_parts))[:, :3]
    return samples[np.random.default_rng(0).permutation(num_parts)]


def _to_plot(points: np.ndarray) -> np.ndarray:
    """Model ``(x, y up, z forward)`` coordinates ``(..., 3)`` as matplotlib's ``(x, depth, up)``."""
    return points[..., [0, 2, 1]]


def _render(triangles: np.ndarray, face_rgb: np.ndarray, path: Path, *, wide: bool) -> None:
    """Draw shaded triangles ``(F, 3, 3)`` (plot space) coloured ``face_rgb (F, 3)`` from a fixed rear camera.

    ``wide`` widens the figure for the exploded views, which spread sideways.
    """
    edges = triangles[:, 1:, :] - triangles[:, :1, :]
    normals = np.cross(edges[:, 0], edges[:, 1])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 0)
    facecolors = np.clip(face_rgb * (0.4 + 0.6 * np.abs(normals @ _LIGHT))[:, None], 0.0, 1.0)

    figure = plt.figure(figsize=(6.5 if wide else 4.5, 6.5))
    axes = figure.add_subplot(111, projection="3d")
    axes.add_collection3d(
        Poly3DCollection(triangles, facecolors=facecolors, edgecolors=(0, 0, 0, 0.08), linewidths=0.1)
    )
    lower, upper = triangles.reshape(-1, 3).min(axis=0), triangles.reshape(-1, 3).max(axis=0)
    axes.set_xlim(lower[0], upper[0])
    axes.set_ylim(lower[1], upper[1])
    axes.set_zlim(lower[2], upper[2])
    axes.set_box_aspect(upper - lower)
    axes.view_init(elev=8.0, azim=-90.0)
    axes.set_axis_off()
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"  wrote {path}")


def _surface(vertices: np.ndarray, faces: np.ndarray, labels: torch.Tensor, colors: np.ndarray) -> tuple:
    """Triangles and colours of the intact mesh, each face in the colour of the part it joins (grey: unlabelled)."""
    face_labels = PartTopology.face_parts(torch.from_numpy(faces), labels).numpy()
    face_rgb = np.where(face_labels[:, None] >= 0, colors[np.clip(face_labels, 0, None)], _UNLABELLED)
    return _to_plot(vertices[faces]), face_rgb


def _split(meshes: Meshes, colors: np.ndarray, centroid: np.ndarray) -> tuple:
    """Triangles and colours of the part meshes ``(vertices, faces)``, each pushed out from the body ``centroid``.

    An open shell may be empty: a labelled part can hold no face of its own.
    """
    triangles, face_rgb = [], []
    for color, (vertices, faces) in zip(colors, meshes, strict=True):
        if len(faces) == 0:
            continue
        exploded = vertices + _EXPLODE_GAIN * (vertices.mean(axis=0) - centroid)
        triangles.append(_to_plot(exploded[faces]))
        face_rgb.append(np.broadcast_to(color, (len(faces), 3)))
    return np.concatenate(triangles), np.concatenate(face_rgb)


def _closed_meshes(topology: PartTopology, vertices: torch.Tensor, joints: torch.Tensor) -> Meshes:
    """The closed part meshes of ``topology`` on the rest mesh ``vertices (V, 3)`` and joints ``joints (J, 3)``."""
    return [(part.numpy(), faces.numpy()) for part, faces in topology.meshes(vertices, joints)]


def _open_meshes(vertices: np.ndarray, faces: np.ndarray, labels: torch.Tensor, num_parts: int) -> Meshes:
    """The open surface shell of each labelled part: the faces it joins, over the vertices they use."""
    face_labels = PartTopology.face_parts(torch.from_numpy(faces), labels).numpy()
    meshes = []
    for part in range(num_parts):
        part_faces = faces[face_labels == part]
        used = np.unique(part_faces)
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        meshes.append((vertices[used], remap[part_faces]))
    return meshes


def _shipped_labels(family: str, num_vertices: int) -> tuple[torch.Tensor, int]:
    """The studio's shipped per-vertex labelling of ``family``: ``(labels (V,), number of parts)``, ``-1`` unlabelled.

    Parts are numbered in sorted name order.
    """
    asset = resources.files("better_human.studio.assets") / f"{family}_vert_segmentation.json"
    parts = bh.io.upgrade(json.loads(asset.read_text()), format="better_human.studio.segmentation", source=str(asset))
    labels = torch.full((num_vertices,), -1, dtype=torch.long)
    for index, name in enumerate(sorted(parts["parts"])):
        labels[parts["parts"][name]] = index
    return labels, len(parts["parts"])


def _render_family(family: str, human: bh.HumanModel, out_dir: Path) -> None:
    """Write the argmax images of ``family`` and, when it ships a per-vertex labelling, the authored ones."""
    spec = human.spec
    identity = human.bake()
    vertices, joints = identity.rest_vertices.detach().cpu(), identity.rest_joint_translations.detach().cpu()
    faces = spec.faces.cpu().numpy()
    points = vertices.numpy()
    centroid = points.mean(axis=0)

    part_ids = spec.parts.ids.cpu()
    colors = _colors(spec.parts.count)
    argmax = PartTopology.from_part_ids(spec.faces.cpu(), part_ids, num_parts=spec.parts.count)
    _render(*_surface(points, faces, part_ids, colors), out_dir / f"{family}_argmax_surface.png", wide=False)
    _render(
        *_split(_closed_meshes(argmax, vertices, joints), colors, centroid),
        out_dir / f"{family}_argmax_split.png",
        wide=True,
    )
    if family not in ("smpl", "smplx"):
        return

    labels, num_parts = _shipped_labels(family, len(points))
    colors = _colors(num_parts)
    _render(*_surface(points, faces, labels, colors), out_dir / f"{family}_authored_surface.png", wide=False)
    if spec.parts.closed_faces is not None:
        meshes = _closed_meshes(human.part_topology, vertices, joints)
        colors = _colors(len(meshes))
    else:
        meshes = _open_meshes(points, faces, labels, num_parts)
    _render(*_split(meshes, colors, centroid), out_dir / f"{family}_authored_split.png", wide=True)


def main() -> None:
    """Render the images of every family whose weights are present."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DOCS_IMAGES, help="where to write (default: docs/segmentation)")
    out_dir = parser.parse_args().out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        try:
            human = bh.load(family)
        except bh.io.AssetNotFoundError as error:
            print(f"skipping {family}: {error}")
            continue
        print(f"rendering {family}")
        with torch.no_grad():
            _render_family(family, human, out_dir)


if __name__ == "__main__":
    main()
