# Frame set (`better_human.frame_set`)

Named mesh points, each a vertex id and an optional parent joint: the input of `Body(contact_frames=...)`. A UTF-8
JSON file, read by `better_human.io.read_frame_set(path)` and written by `better_human.io.write_frame_set(path,
frames)`; BetterHuman ships the SMPL-X contact set as `better_human.io.shipped_frame_set("smplx_contacts")`.

| version | status | form |
|---|---|---|
| 1 | current | header and `frames` |
| 0 | read through migration | the bare mapping `{name: vertex | {"vertex", "joint"}}`, no header |

## Version 1

```json
{
  "format": "better_human.frame_set",
  "format_version": 1,
  "frames": {
    "heel_left": {"vertex_id": 8924, "joint_name": null},
    "knee_left": {"vertex_id": 3673, "joint_name": "left_knee"}
  }
}
```

| key | type | meaning |
|---|---|---|
| `format` | string | `"better_human.frame_set"` |
| `format_version` | integer | `1` |
| `frames` | object | frame name to entry, in authored order |
| `frames.<name>.vertex_id` | integer >= 0 | mesh vertex carrying the frame; range-checked against the mesh when the frames are attached to a body |
| `frames.<name>.joint_name` | string or null | parent joint name; `null` takes the vertex's dominant skinning joint |

The reader ignores a key it does not know, at the top level and inside an entry; the writer writes only the keys
above. The writer uses two-space indentation and a trailing newline, and replaces the file atomically.

## Migration

`better_human.io.formats` migrates version 0 in memory: it wraps the mapping in the header and turns a bare vertex id
into `{"vertex_id": id, "joint_name": null}`, and an entry `{"vertex", "joint"}` into `{"vertex_id", "joint_name"}`. A file is version 0 only when it has no `format` key and every value is a vertex
id or `{"vertex", "joint"}`; any other file without `format`, a top-level array, and a file with `format` but no
`format_version` are refused. A later version is refused with both versions named.
