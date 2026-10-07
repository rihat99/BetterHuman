"""Render the segmentation comparison images embedded in ``docs/concepts/segmentation.md``.

For every available body (SMPL, SMPL-X, MHR) and every segmentation type it supports
(authored, argmax) this writes a part-colored surface and an exploded body-split
view into ``docs/segmentation/``. Rendering is headless matplotlib (Agg) with a
fixed camera, fixed colors, and no randomness beyond a seeded color permutation,
so re-runs are byte-stable.

Regenerate with::

    uv run python docs/render_segmentation.py

Weights resolve through the normal loader: pass ``--models-dir`` or set
``$BETTERHUMAN_MODELS_DIR``. Any model whose weights are absent is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import better_human as bh  # noqa: E402
from better_human.bodies.assets import _ASSET_DIR  # noqa: E402
from better_human.core.mass import build_part_topology, meshes_from_topology  # noqa: E402

_OUT_DIR = Path(__file__).parent / "segmentation"
_LIGHT = np.array([0.3, -0.8, 0.5])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)
_EXPLODE_GAIN = 0.55


def _colors(num_parts: int) -> np.ndarray:
    """Return ``num_parts`` distinct RGB colors ordered for adjacent contrast."""
    samples = matplotlib.colormaps["gist_rainbow"](np.linspace(0.0, 1.0, num_parts))[:, :3]
    order = np.random.default_rng(0).permutation(num_parts)
    return samples[order]


def _to_plot(coords: np.ndarray) -> np.ndarray:
    """Reorder model ``(x, y_up, z_forward)`` to matplotlib ``(x, depth, up)``."""
    return coords[..., [0, 2, 1]]


def _render(triangles: np.ndarray, face_rgb: np.ndarray, path: Path, *, wide: bool) -> None:
    """Rasterize shaded triangles from a fixed rear camera to ``path``.

    :param triangles: plot-space triangle vertices ``(F, 3, 3)``.
    :param face_rgb: base RGB per triangle ``(F, 3)``.
    :param wide: widen the figure for the laterally spread split views.
    """
    edges = triangles[:, 1:, :] - triangles[:, :1, :]
    normals = np.cross(edges[:, 0], edges[:, 1])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 0)
    shade = 0.4 + 0.6 * np.abs(normals @ _LIGHT)
    facecolors = np.clip(face_rgb * shade[:, None], 0.0, 1.0)

    figure = plt.figure(figsize=(6.5 if wide else 4.5, 6.5))
    axes = figure.add_subplot(111, projection="3d")
    axes.add_collection3d(
        Poly3DCollection(triangles, facecolors=facecolors, edgecolors=(0, 0, 0, 0.08), linewidths=0.1)
    )
    lower = triangles.reshape(-1, 3).min(axis=0)
    upper = triangles.reshape(-1, 3).max(axis=0)
    axes.set_xlim(lower[0], upper[0])
    axes.set_ylim(lower[1], upper[1])
    axes.set_zlim(lower[2], upper[2])
    axes.set_box_aspect(upper - lower)
    axes.view_init(elev=8.0, azim=-90.0)
    axes.set_axis_off()
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"  wrote {path.relative_to(Path.cwd())}")


def _majority_labels(faces: np.ndarray, vertex_labels: np.ndarray) -> np.ndarray:
    """Label each face by the part held by at least two of its vertices."""
    labels = vertex_labels[faces]
    first, second, third = labels[:, 0], labels[:, 1], labels[:, 2]
    pick_first = (first == second) | (first == third)
    return np.where(pick_first, first, np.where(second == third, second, first))


def _vertseg_labels(filename: str, num_vertices: int) -> tuple[np.ndarray, int]:
    """Load an authored per-vertex segmentation as dense integer labels.

    :returns: ``(labels (V,), num_parts)`` with unlabeled vertices set to ``-1``.
    """
    named_parts = json.loads((_ASSET_DIR / filename).read_text())
    labels = np.full(num_vertices, -1, dtype=np.int64)
    for index, part_name in enumerate(sorted(named_parts)):
        labels[named_parts[part_name]] = index
    return labels, len(named_parts)


def _surface_triangles(vertices: np.ndarray, faces: np.ndarray, face_labels: np.ndarray, colors: np.ndarray):
    """Build intact-surface triangles colored by per-face part label."""
    unlabeled = np.array([0.8, 0.8, 0.8])
    face_rgb = np.where(face_labels[:, None] >= 0, colors[np.clip(face_labels, 0, None)], unlabeled)
    return _to_plot(vertices[faces]), face_rgb


def _explode_part(vertices: np.ndarray, body_centroid: np.ndarray) -> np.ndarray:
    """Translate one part's vertices radially outward from the body centroid."""
    return vertices + _EXPLODE_GAIN * (vertices.mean(axis=0) - body_centroid)


def _split_triangles(part_meshes, colors: np.ndarray, body_centroid: np.ndarray):
    """Build exploded triangles from a list of ``(vertices, faces)`` part meshes."""
    triangles: list[np.ndarray] = []
    face_rgb: list[np.ndarray] = []
    for index, (vertices, faces) in enumerate(part_meshes):
        if faces.shape[0] == 0:
            continue
        exploded = _explode_part(vertices, body_centroid)
        part_triangles = _to_plot(exploded[faces])
        triangles.append(part_triangles)
        face_rgb.append(np.broadcast_to(colors[index], (part_triangles.shape[0], 3)))
    return np.concatenate(triangles), np.concatenate(face_rgb)


def _topology_meshes(v_shaped: torch.Tensor, topology, rest_joints: torch.Tensor):
    """Detach closed part meshes to numpy ``(vertices, faces)`` pairs."""
    return [
        (vertices.detach().cpu().numpy(), faces.detach().cpu().numpy())
        for vertices, faces in meshes_from_topology(v_shaped, topology, rest_joints)
    ]


def _vertseg_meshes(vertices: np.ndarray, faces: np.ndarray, vertex_labels: np.ndarray, num_parts: int):
    """Split the intact mesh into per-part open shells by vertex labels."""
    face_labels = _majority_labels(faces, vertex_labels)
    meshes: list[tuple[np.ndarray, np.ndarray]] = []
    for part in range(num_parts):
        part_faces = faces[face_labels == part]
        used = np.unique(part_faces)
        remap = np.full(vertices.shape[0], -1, dtype=np.int64)
        remap[used] = np.arange(used.size)
        meshes.append((vertices[used], remap[part_faces]))
    return meshes


def _render_body(family: str, body) -> None:
    """Render both segmentation types (authored, argmax) for one loaded body."""
    vertices = body.values.v_shaped.detach().cpu().numpy()
    faces = body.structure.faces.detach().cpu().numpy()
    body_centroid = vertices.mean(axis=0)
    part_ids = body.structure.part_ids.detach().cpu().numpy()

    argmax_parts = int(part_ids.max()) + 1
    argmax_colors = _colors(argmax_parts)
    argmax_surface = _surface_triangles(vertices, faces, _majority_labels(faces, part_ids), argmax_colors)
    _render(*argmax_surface, _OUT_DIR / f"{family}_argmax_surface.png", wide=False)
    argmax_topology = build_part_topology(body.structure.faces, body.structure.part_ids, argmax_parts)
    argmax_meshes = _topology_meshes(body.values.v_shaped, argmax_topology, body.values.rest_joints)
    _render(*_split_triangles(argmax_meshes, argmax_colors, body_centroid),
            _OUT_DIR / f"{family}_argmax_split.png", wide=True)

    vertseg_name = {"smpl": "smpl_vert_segmentation.json", "smplx": "smplx_vert_segmentation.json"}.get(family)
    if vertseg_name is None:
        return
    vertex_labels, authored_parts = _vertseg_labels(vertseg_name, vertices.shape[0])
    authored_colors = _colors(authored_parts)
    authored_surface = _surface_triangles(vertices, faces, _majority_labels(faces, vertex_labels), authored_colors)
    _render(*authored_surface, _OUT_DIR / f"{family}_authored_surface.png", wide=False)

    if body.structure.part_topology is not None and family == "smpl":
        closed_meshes = _topology_meshes(body.values.v_shaped, body.structure.part_topology, body.values.rest_joints)
        closed_colors = _colors(len(closed_meshes))
        _render(*_split_triangles(closed_meshes, closed_colors, body_centroid),
                _OUT_DIR / f"{family}_authored_split.png", wide=True)
    else:
        open_meshes = _vertseg_meshes(vertices, faces, vertex_labels, authored_parts)
        _render(*_split_triangles(open_meshes, authored_colors, body_centroid),
                _OUT_DIR / f"{family}_authored_split.png", wide=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, help="Weight root; else use $BETTERHUMAN_MODELS_DIR")
    return parser.parse_args()


def main() -> None:
    """Render every available body's segmentation images, skipping absent weights."""
    arguments = _arguments()
    if arguments.models_dir is not None:
        os.environ["BETTERHUMAN_MODELS_DIR"] = str(arguments.models_dir)
    _OUT_DIR.mkdir(exist_ok=True)
    loaders = (
        ("smpl", bh.SMPL, {"gender": "neutral"}),
        ("smplx", bh.SMPLX, {"gender": "neutral"}),
        (
            "mhr",
            bh.MHR,
            {
                "lod": 1,
                "use_expression": False,
                "use_correctives": False,
                "compute_mass": False,
            },
        ),
    )
    for family, loader, kwargs in loaders:
        try:
            body = loader(**kwargs)
        except FileNotFoundError as error:
            print(f"skipping {family}: {error}")
            continue
        print(f"rendering {family}")
        _render_body(family, body)


if __name__ == "__main__":
    main()
