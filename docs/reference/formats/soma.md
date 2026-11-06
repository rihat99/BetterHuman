# SOMA archive (`better_human.soma`)

The converted SOMA-X mid level of detail, `$BETTERHUMAN_MODELS_DIR/SOMA/soma_mid.npz`, written by
`tools/convert/soma/convert_soma.py` from the SOMA-X release assets and read by `better_human.families.soma`. A
NumPy `.npz` archive, read with `allow_pickle=False`.

| version | status | written by |
|---|---|---|
| 2 | current | `convert_soma.py` from decision record 0012 on |
| 1 | read through migration | `convert_soma.py` before record 0012 (no `format`; its provenance JSON was `source`) |

## Header

| field | dtype | shape | meaning |
|---|---|---|---|
| `format` | str | `()` | `"better_human.soma"` |
| `format_version` | int64 | `()` | `2` |
| `provenance` | str | `()` | JSON: `converter`, `package` (`py-soma-x`), `package_version`, `asset_release_tag`, `asset_commit`, `sha256` of every asset read |

## Fields the reader reads

The sizes are fixed: `V = 18056` vertices, `S = 128` shape components, `J = 110` skin joints, `P = 78` public
joints, `T = 32` twist helpers; `F`, `K`, `C`, `H` and `N` follow the file. Transforms are 4x4 world matrices.

| field | dtype | shape | meaning |
|---|---|---|---|
| `unit` | `<U1` | `()` | `"m"` |
| `up_axis`, `forward_axis` | `<U2` | `()` | `"+y"`, `"+z"` |
| `shape_mean`, `bind_vertices` | float32 | `(V, 3)` | mean shape, bind mesh |
| `shape_basis` | float32 | `(V, 3, S)` | identity blend shapes |
| `faces` | int64 | `(F, 3)` | triangles |
| `joint_names` | `<U18` | `(J,)` | skin joints |
| `joint_parents` | int64 | `(J,)` | parent per joint, `-1` for the root |
| `bind_pose_world`, `t_pose_world` | float32 | `(J, 4, 4)` | bind pose and T-pose |
| `skin_indices`, `skin_weights` | int64, float32 | `(V, K)` each | compact skinning, rows sum to 1 |
| `public_joint_ids`, `public_parents` | int64 | `(P,)` each | the posable joints and their parents |
| `public_skin_weights` | float32 | `(V, P)` | skinning onto the public joints |
| `joint_regressor` | float32 | `(P, V)` | public joint positions from the mesh |
| `twist_joint_ids`, `twist_segment` | int64 | `(T,)` each | twist helpers and their limb segment |
| `twist_fraction` | float32 | `(T,)` | position of each helper along its segment |
| `twist_rotation_matrix` | float32 | `(T, P)` | helper twist from the public rotations |
| `segment_start`, `segment_end`, `segment_parent` | int64 | `(8,)` each | the eight limb segments |
| `segment_reverse` | bool | `(8,)` | segments whose start joint also takes the segment's twist |
| `corrective_bind` | float32 | `(P, 3, 3)` | local bind rotations of the corrective MLP |
| `corrective_first_layer` | float32 | `(C, H)` | first layer of the pose-corrective MLP |
| `corrective_output_indices`, `corrective_output_values` | int64, float32 | `(2, N)`, `(N,)` | its sparse output layer |
| `corrective_output_shape` | int64 | `(2,)` | the output layer's dense shape |
| `corrective_tanh` | bool | `()` | must be false |

The converter also writes `lod = "mid"`; the reader ignores every field it does not list here. `correctives=False`
skips `corrective_first_layer` and the output layer.

## Migration

`better_human.io.formats` migrates a v1 archive in memory: it adds `format`, `provenance = "unknown (v1)"` and
`format_version = 2`, so v1 files on disk stay valid (a v1 `source` field is ignored). A later version is refused with
both versions named.
