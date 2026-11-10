# Density tables (`better_human.density_tables`)

Named tables of body-segment densities in kg/m³, keyed by density category: what `density="Dempster"` reads. BetterHuman
ships one archive, `src/better_human/io/assets/density_tables.npz`, read by
`better_human.io.mass_data.shipped_density_tables()`; each family loader puts it in `Parts.density_tables` and names
the category of each part in `Parts.density_categories`. A NumPy `.npz` archive, read with `allow_pickle=False`.

| version | status | written by |
|---|---|---|
| 1 | current | converted from `body_densities.json` (keyed `"0"`..`"23"`), shipped before decision record 0013 |

## Header

| field | dtype | shape | meaning |
|---|---|---|---|
| `format` | str | `()` | `"better_human.density_tables"` |
| `format_version` | int64 | `()` | `1` |
| `provenance` | str | `()` | JSON: the file it was converted from and the commit |

## Fields the reader reads

`T` is the number of tables (shipped: `Naive`, `Chandler`, `Dempster`, `Clauser`, `Custom`) and `C` the number of
categories (shipped: SMPL's 24 joint names, `pelvis` to `right_hand`).

| field | dtype | shape | meaning |
|---|---|---|---|
| `table_names` | str | `(T,)` | the name `density=` takes |
| `categories` | str | `(C,)` | density categories |
| `densities` | float64 | `(T, C)` | `densities[t, c]`: table `t`'s density of category `c`, kg/m³ |

## Fields the reader does not read

`sources` (str, `(T, C)`) says where `densities[t, c]` comes from, in one of three forms: a publication, its table and
the segment, for a number read in that publication; `unverified (cited from memory): <citation>` for a citation not
yet checked against the publication; or `assumed: <reason>` for a value no publication gives. Every shipped density
names its source (`tests/families/test_density_categories.py`); `shipped_density_tables()` does not return it.

The reader ignores a field it does not know. A later version is refused with both versions named.
