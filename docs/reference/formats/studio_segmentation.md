# Studio segmentation (`better_human.studio.segmentation`)

A vertex segmentation authored in the studio: each part name with the mesh vertices painted into it. A UTF-8 JSON
file under the studio's data root (`<data root>/segmentations/<model key>/<name>.json`), read by
`better_human.studio.persistence.read_segmentation(path)` and written by `write_segmentation(path, parts)`. The
studio ships the SMPL and SMPL-X segmentations in this format (`better_human/studio/assets/`).

| version | status | form |
|---|---|---|
| 1 | current | header and `parts` |
| 0 | read through migration | the bare mapping `{part name: [vertex ids]}` the studio wrote before its files named a format, no header |

## Version 1

```json
{
  "format": "better_human.studio.segmentation",
  "format_version": 1,
  "parts": {
    "head": [0, 1, 2],
    "torso": [3, 4]
  }
}
```

| key | type | meaning |
|---|---|---|
| `format` | string | `"better_human.studio.segmentation"` |
| `format_version` | integer | `1` |
| `parts` | object | part name to its vertex ids, in authored order |
| `parts.<name>` | non-empty array of integers >= 0 | the part's mesh vertices; range-checked against the mesh when the studio applies the file |

The reader ignores a top-level key it does not know; the writer writes only the keys above, with two-space
indentation and a trailing newline. A file of another format, a later version, or a part that is not a non-empty
array of vertex ids is refused with `better_human.io.FormatError`, whose message ends "re-save it in the studio".

## Migration

`better_human.io.formats` migrates version 0 in memory: it wraps the mapping in the header as `parts`. A file is
version 0 only when it has no `format` key and every value is an array; any other file without `format`, and a file
that is not a JSON object, are refused.
