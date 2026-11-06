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
    "heel_left": {"vertex": 8924, "joint": null},
    "knee_left": {"vertex": 3673, "joint": "left_knee"}
  }
}
```

| key | type | meaning |
|---|---|---|
| `format` | string | `"better_human.frame_set"` |
| `format_version` | integer | `1` |
| `frames` | object | frame name to entry, in authored order |
| `frames.<name>.vertex` | integer | mesh vertex carrying the frame; range-checked against the mesh when the frames are attached to a body |
| `frames.<name>.joint` | string or null | parent joint name; `null` takes the vertex's dominant skinning joint |

The writer uses two-space indentation and a trailing newline, and replaces the file atomically.

## Migration

`better_human.io.formats` migrates version 0 in memory: it wraps the mapping in the header and turns a bare vertex id
into `{"vertex": id, "joint": null}`. A later version is refused with both versions named.
