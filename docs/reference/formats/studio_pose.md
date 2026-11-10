# Studio pose (`better_human.studio.pose`)

A configuration saved in the studio, with the names of the model joints it follows. A UTF-8 JSON file under the
studio's data root (`<data root>/poses/<model key>/<name>.json`), read by
`better_human.studio.persistence.read_pose(path)` and written by `write_pose(path, pose)`.

| version | status | form |
|---|---|---|
| 1 | current | header, `joint_names` and `q` |

## Version 1

```json
{
  "format": "better_human.studio.pose",
  "format_version": 1,
  "joint_names": ["root", "spine"],
  "q": [0.0, 0.5, -0.25]
}
```

| key | type | meaning |
|---|---|---|
| `format` | string | `"better_human.studio.pose"` |
| `format_version` | integer | `1` |
| `joint_names` | array of strings | the BetterRobot model's `joint_names`, in the order `q` follows them |
| `q` | array of numbers | the configuration, `nq` values in BetterRobot's layout (pose `[tx, ty, tz, qx, qy, qz, qw]` for a free flyer) |

The studio loads a pose only onto a model with the same `joint_names`; any other model is refused naming the file.
The reader ignores a top-level key it does not know; the writer writes only the keys above, with two-space
indentation and a trailing newline. A file of another format or a later version is refused with
`better_human.io.FormatError`. There is no earlier version to migrate.
