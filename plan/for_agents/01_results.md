# Order 01 results — end-to-end API and body-family layout

Status: complete on `dev`, 2026-07-22. The implementation is behavior-
preserving: every recorded parity maximum is byte-identical before and after.

## Implementation

- Added the specified `ParametricBody.__call__`: no shape arguments delegates
  to `vertices(q)`; named shape groups delegate through `with_shape` first.
- Added four float32 tests covering exact SMPL and MHR one-call equality,
  validation passthrough, and exact CUDA equality.
- Kept the public root API and all `better_human.bodies.*Classic` exports
  unchanged. No compatibility shims were added for private alpha paths.
- Documented one-off and fixed-shape usage, plus the verified composition-over-
  inheritance decision in `docs/architecture.md`.

## Move map

| Before | After |
|---|---|
| `bodies/smpl.py` | `bodies/smpl_family/smpl.py` |
| `bodies/smplx.py` | `bodies/smpl_family/smplx.py` |
| `bodies/smpl_family.py` | `bodies/smpl_family/shared.py` |
| `bodies/config_smpl.py` | `bodies/smpl_family/config_smpl.py` |
| `bodies/config_smplx.py` | `bodies/smpl_family/config_smplx.py` |
| `bodies/config_mhr.py` | `bodies/mhr/config.py` |
| `bodies/mhr.py` archive access/validation | `bodies/mhr/archive.py` |
| `bodies/mhr.py` archive-to-robot binding | `bodies/mhr/tree.py` |
| `bodies/mhr.py` shape bake/corrective | `bodies/mhr/deformation.py` |
| `bodies/mhr.py` loader/classic boundary | `bodies/mhr/body.py` |

The tracked source files were moved with `git mv`. New family `__init__.py`
files provide the planned re-exports. SMPL-family function bodies were not
changed; only import paths changed.

## Lines of code

Recursive `wc -l` under `src/better_human/bodies`, including the unchanged
asset facade:

### Before

| Module | LOC |
|---|---:|
| `bodies/__init__.py` | 8 |
| `bodies/assets/__init__.py` | 66 |
| `bodies/config_body.py` | 49 |
| `bodies/config_mhr.py` | 36 |
| `bodies/config_smpl.py` | 89 |
| `bodies/config_smplx.py` | 124 |
| `bodies/mhr.py` | 762 |
| `bodies/registry.py` | 40 |
| `bodies/smpl.py` | 184 |
| `bodies/smpl_family.py` | 265 |
| `bodies/smplx.py` | 389 |
| **Total** | **2,012** |

### After

| Module | LOC |
|---|---:|
| `bodies/__init__.py` | 7 |
| `bodies/assets/__init__.py` | 66 |
| `bodies/config_body.py` | 49 |
| `bodies/registry.py` | 40 |
| `bodies/smpl_family/__init__.py` | 6 |
| `bodies/smpl_family/config_smpl.py` | 89 |
| `bodies/smpl_family/config_smplx.py` | 124 |
| `bodies/smpl_family/shared.py` | 265 |
| `bodies/smpl_family/smpl.py` | 184 |
| `bodies/smpl_family/smplx.py` | 389 |
| `bodies/mhr/__init__.py` | 5 |
| `bodies/mhr/archive.py` | 161 |
| `bodies/mhr/body.py` | 319 |
| `bodies/mhr/config.py` | 36 |
| `bodies/mhr/deformation.py` | 120 |
| `bodies/mhr/tree.py` | 226 |
| **Total** | **2,086** |

The largest new MHR module is 319 lines (approximately the 300-line target).
The only materially larger module is the unchanged 389-line SMPL-X loader,
the exception allowed by the order.

## Exact parity maxima

The same read-only reporter was used before and after (SHA-256
`bf075cbe744df2e3bc52f7bb39ec56e7007a9cd88d32a1582434eace22986e73`).

### SMPL family — before and after

| Model | dtype | V before | V after | J before | J after |
|---|---|---:|---:|---:|---:|
| SMPL | float32 | 7.059425115585327e-07 | 7.059425115585327e-07 | 5.289912223815918e-07 | 5.289912223815918e-07 |
| SMPL | float64 | 1.0412031081763473e-07 | 1.0412031081763473e-07 | 5.780450140768778e-08 | 5.780450140768778e-08 |
| SMPL-X | float32 | 6.258487701416016e-07 | 6.258487701416016e-07 | 5.066394805908203e-07 | 5.066394805908203e-07 |
| SMPL-X | float64 | 5.590597473847936e-08 | 5.590597473847936e-08 | 8.653926908408494e-09 | 8.653926908408494e-09 |

### MHR — before

`V/P/Q/S` are maximum vertex, joint-position, sign-invariant quaternion,
and accumulated-scale errors.

| dtype | LOD | V | P | Q | S |
|---|---:|---:|---:|---:|---:|
| float32 | 0 | 2.0265579223632812e-06 | 3.5762786865234375e-07 | 2.980232238769531e-07 | 0.0 |
| float32 | 1 | 2.8014183044433594e-06 | 7.599592208862305e-07 | 6.556510925292969e-07 | 0.0 |
| float32 | 2 | 1.9669532775878906e-06 | 3.5762786865234375e-07 | 2.980232238769531e-07 | 0.0 |
| float32 | 3 | 2.2649765014648438e-06 | 7.152557373046875e-07 | 4.76837158203125e-07 | 0.0 |
| float32 | 4 | 2.0265579223632812e-06 | 3.5762786865234375e-07 | 3.5762786865234375e-07 | 0.0 |
| float32 | 5 | 1.9073486328125e-06 | 4.76837158203125e-07 | 2.980232238769531e-07 | 0.0 |
| float32 | 6 | 1.7881393432617188e-06 | 2.682209014892578e-07 | 3.5762786865234375e-07 | 0.0 |
| float64 | 0 | 6.796320864976835e-07 | 1.8331952966743614e-07 | 1.3139750559831853e-07 | 2.4801335873902985e-08 |
| float64 | 1 | 8.488698018949137e-07 | 2.215700635188611e-07 | 1.9291339442562006e-07 | 4.6629366146078155e-08 |
| float64 | 2 | 5.581134645549213e-07 | 1.315468958207333e-07 | 1.022059766819794e-07 | 5.7916624163212305e-08 |
| float64 | 3 | 5.617910434629891e-07 | 9.950705104344593e-08 | 1.0819653151195041e-07 | 2.9032722226318697e-08 |
| float64 | 4 | 5.530133108111457e-07 | 2.0618963714191807e-07 | 1.3338477658497716e-07 | 2.5566294747925156e-08 |
| float64 | 5 | 4.4630069462314736e-07 | 1.226731705106232e-07 | 1.2220955582231685e-07 | 2.2953858813679062e-08 |
| float64 | 6 | 5.935375642085461e-07 | 1.2023582485198858e-07 | 1.8299445425640215e-07 | 2.9279848323682245e-08 |

### MHR — after

| dtype | LOD | V | P | Q | S |
|---|---:|---:|---:|---:|---:|
| float32 | 0 | 2.0265579223632812e-06 | 3.5762786865234375e-07 | 2.980232238769531e-07 | 0.0 |
| float32 | 1 | 2.8014183044433594e-06 | 7.599592208862305e-07 | 6.556510925292969e-07 | 0.0 |
| float32 | 2 | 1.9669532775878906e-06 | 3.5762786865234375e-07 | 2.980232238769531e-07 | 0.0 |
| float32 | 3 | 2.2649765014648438e-06 | 7.152557373046875e-07 | 4.76837158203125e-07 | 0.0 |
| float32 | 4 | 2.0265579223632812e-06 | 3.5762786865234375e-07 | 3.5762786865234375e-07 | 0.0 |
| float32 | 5 | 1.9073486328125e-06 | 4.76837158203125e-07 | 2.980232238769531e-07 | 0.0 |
| float32 | 6 | 1.7881393432617188e-06 | 2.682209014892578e-07 | 3.5762786865234375e-07 | 0.0 |
| float64 | 0 | 6.796320864976835e-07 | 1.8331952966743614e-07 | 1.3139750559831853e-07 | 2.4801335873902985e-08 |
| float64 | 1 | 8.488698018949137e-07 | 2.215700635188611e-07 | 1.9291339442562006e-07 | 4.6629366146078155e-08 |
| float64 | 2 | 5.581134645549213e-07 | 1.315468958207333e-07 | 1.022059766819794e-07 | 5.7916624163212305e-08 |
| float64 | 3 | 5.617910434629891e-07 | 9.950705104344593e-08 | 1.0819653151195041e-07 | 2.9032722226318697e-08 |
| float64 | 4 | 5.530133108111457e-07 | 2.0618963714191807e-07 | 1.3338477658497716e-07 | 2.5566294747925156e-08 |
| float64 | 5 | 4.4630069462314736e-07 | 1.226731705106232e-07 | 1.2220955582231685e-07 | 2.2953858813679062e-08 |
| float64 | 6 | 5.935375642085461e-07 | 1.2023582485198858e-07 | 1.8299445425640215e-07 | 2.9279848323682245e-08 |

## Verification

| Gate | Result |
|---|---|
| Pre-change collection | 226 tests |
| Post-change collection | 230 tests |
| Full suite, RTX 6000 Ada GPU 7 | 230 passed, 0 skipped, 19 warnings in 247.38 s |
| Ruff | `ruff check .` clean |
| Patch hygiene | `git diff --check` clean |
| Public imports | root API and all body-facade classic types import successfully |
| MHR CPU smoke | batch 1, 0 warmups, 1 iteration; exit 0 |
| SMPL/SMPL-X CPU smoke | batch 1, 0 warmups, 1 iteration; exit 0; agreement gate passed |

The smoke timings are intentionally not interpreted as performance results;
fair timing and gradient workloads belong to Order 02.

## Deviations and corrected findings

1. The prescribed `mhr/body.py` contents would exceed 400 lines while the
   same order requires modules near 300 lines. Shape bake and pose corrective
   moved verbatim to one cohesive `mhr/deformation.py`; this adds a module,
   not an abstraction. `body.py` is 319 lines. `classic.py` was omitted as the
   plan directed because `MHRClassic` is only a dataclass.
2. BetterRobot source disproves the proposed unbatched-value rationale:
   `ModelValues` validation permits leading batch dimensions. The docs instead
   record the real distinction—different closed schemas, ownership, and update
   lifecycles.
3. An unregistered `ModelValues` subclass is a pytree leaf; it is not normally
   flattened and silently stripped. The docs state the exact-type registration
   and base reconstructor behavior accurately. The alpha-coupling count is
   recorded against the component dataclasses (78 structure + 15 value fields),
   not described as a roughly 70-field `Model` constructor.

No numerical logic, BetterRobot code, model assets, reference code, parity
pins, or benchmark logic changed.
