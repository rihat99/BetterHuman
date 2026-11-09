# Closed segmentation (`better_human.segmentation`)

A family's body cut into closed part meshes, for the part inertias of `HumanModel.bake` (`density=...`). BetterHuman
ships one, `src/better_human/io/assets/smpl_segmentation.npz` (SMPL's 24 parts), read by
`better_human.io.mass_data.shipped_segmentation("smpl")`; the SMPL loader puts it in `Parts.closed_faces`. A NumPy
`.npz` archive, read with `allow_pickle=False`.

| version | status | written by |
|---|---|---|
| 1 | current | converted from the pickled `smpl_3d_segmentation.npy` shipped before decision record 0013 |

## Header

| field | dtype | shape | meaning |
|---|---|---|---|
| `format` | str | `()` | `"better_human.segmentation"` |
| `format_version` | int64 | `()` | `1` |
| `provenance` | str | `()` | JSON: the file it was converted from and the commit |

## Fields the reader reads

`P` is the number of parts. A face indexes the concatenation `[vertices (V, 3); joints (J, 3)]` of the family's rest
mesh and rest joints: an index `>= V` is joint `index - V`, a cap that closes the part where it meets its neighbour.

| field | dtype | shape | meaning |
|---|---|---|---|
| `num_vertices` | int64 | `()` | `V`, the family's vertex count, where joint indices start |
| `faces` | int64 | `(F, 3)` | every part's triangles, part after part |
| `part_offsets` | int64 | `(P+1,)` | part `p` is `faces[part_offsets[p]:part_offsets[p + 1]]`; starts at `0`, ends at `F` |

The reader ignores a field it does not know. A later version is refused with both versions named.
