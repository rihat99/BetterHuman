# MHR archive (`better_human.mhr`)

One converted MHR level of detail, `$BETTERHUMAN_MODELS_DIR/MHR/converted/mhr_lod{N}.npz` (`N` in 0..6), written by
`tools/convert/mhr/convert_mhr.py` from the MHR release assets and read by `better_human.families.mhr`. A NumPy
`.npz` archive, read with `allow_pickle=False`.

| version | status | written by |
|---|---|---|
| 2 | current | `convert_mhr.py` from decision record 0012 on |
| 1 | read through migration | `convert_mhr.py` before record 0012 (no `format`, no `provenance`) |

## Header

| field | dtype | shape | meaning |
|---|---|---|---|
| `format` | str | `()` | `"better_human.mhr"` |
| `format_version` | int64 | `()` | `2` |
| `provenance` | str | `()` | JSON: `converter`, `package` (`pymomentum` or `pymomentum-cpu`), `package_version`, `sha256` of every asset read |

## Fields the reader reads

Lengths are metres, rotations are quaternions `[qx, qy, qz, qw]`, the up axis is `+y` (stored, `up_axis`). `V` and
`F` are the LOD's vertex and face counts.

| field | dtype | shape | meaning |
|---|---|---|---|
| `lod` | int64 | `()` | level of detail; must equal the loader's `lod` |
| `joint_names` | `<U20` | `(127,)` | joint names, parents first |
| `joint_parents` | int64 | `(127,)` | parent joint per joint, `-1` for the root |
| `joint_pre_rotations` | float32 | `(127, 4)` | joint pre-rotation quaternions |
| `joint_translation_offsets` | float32 | `(127, 3)` | joint offsets from the parent |
| `segmentation_group_names` | `<U11` | `(20,)` | body-part groups |
| `segmentation_joint_group_indices` | int64 | `(127,)` | group of each joint |
| `model_parameter_names` | `<U24` | `(204,)` | upstream model parameters |
| `model_parameter_channel_types` | int8 | `(204,)` | `0` translation, `1` rotation, `2` scale |
| `parameter_transform_rows`, `parameter_transform_columns`, `parameter_transform_values` | int64, int64, float32 | `(368,)` each | sparse parameter transform: row `joint * 7 + channel`, column model parameter |
| `model_parameter_lower_limits`, `model_parameter_upper_limits` | float32 | `(204,)` each | parameter limits |
| `template_vertices` | float32 | `(V, 3)` | rest mesh |
| `faces` | int64 | `(F, 3)` | triangles |
| `skin_indices`, `skin_weights` | int64, float32 | `(V, 8)` each | padded skinning, rows sum to 1 |
| `identity_basis` | float32 | `(V, 3, 45)` | identity blend shapes |
| `expression_basis` | float32 | `(V, 3, 72)` | expression blend shapes |
| `up_axis` | `<U2` | `()` | `"+y"`; sets gravity |
| `bind_matrices`, `inverse_bind_matrices` | float32 | `(127, 4, 4)` | bind poses and their inverses |
| `corrective_first_layer_indices` | int64 | `(2, 53136)` | sparse first layer of the pose-corrective MLP |
| `corrective_first_layer_weights` | float32 | `(53136,)` | its values |
| `corrective_first_layer_mask` | bool | `(3000, 750)` | its dense mask |
| `corrective_output_weights` | float32 | `(3000, V * 3)` | the MLP's output layer |

The converter writes further fields (conversion checks, upstream metadata); the reader ignores every field it does
not list here. `correctives=False` and `expression=False` skip the corrective and expression fields.

## Migration

`better_human.io.formats` migrates a v1 archive in memory: it adds `format`, `provenance = "unknown (v1)"` and
`format_version = 2`, so v1 files on disk stay valid. A later version is refused with both versions named.
