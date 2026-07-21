# Order 05 results

Completed on `dev` on 2026-07-20 after Order 04.

## Delivered

- Static argmax-skinning vertex segmentation, deterministic majority face
  ownership, half-edge boundary-cycle tracing, reverse-wound centroid caps,
  and compact per-part topology in `core/mass.py`.
- Batched, differentiable part integration through BetterRobot
  `Inertia.from_mesh`. Physical rows are translated into native rest-joint
  frames, and a massless universe row is prepended for exact BetterRobot body
  indexing.
- Scalar density `1000.0` kg/m^3 by default, or a positive scalar/per-part
  tensor override in native joint order. The inspected data file also offers
  `Naive`, `Chandler`, `Dempster`, `Clauser`, and `Custom` tables; these feed
  the same tensor path rather than separate policy code.
- Shape baking now updates both `BodyValues.body_inertias` and
  `ModelValues.body_inertias`. Full and reduced SMPL-X spawns use the same
  model-agnostic path as SMPL.
- `compute_mass=True` is the spawn default. The evidence-backed
  `compute_mass=False` option skips topology/inertia work for deformation-only
  workloads; deformation parity and converter tests use it deliberately.
- Closedness, conservation, COM, density, float32 gradcheck, batched RNEA/CRBA,
  shape-to-RNEA backward, SMPL-X full/partial, and CUDA coverage. Architecture
  and units/contracts documentation were updated in the same order.

## Neutral SMPL measurements

All masses use uniform density `1000.0` kg/m^3.

| Quantity | Measured value |
|---|---:|
| Sum of 24 closed parts | `73.43093872070312` kg |
| Whole closed mesh | `74.76054382324219` kg |
| Absolute partition gap | `1.3296051025390625` kg |
| Relative partition gap | `0.01778485059823366` |
| Part-aggregate COM | `[0.0006243017851375043, 0.055163439363241196, -0.020244255661964417]` m |
| Whole-mesh COM | `[0.0006061043823137879, 0.05762733519077301, -0.020352846011519432]` m |
| COM separation | `0.0024663547308557813` m |
| Centered mesh bounds | `[-0.8701626062393188, -0.9385096430778503, -0.15066076815128326]` to `[0.8749125003814697, 0.7788512110710144, 0.13980798423290253]` m |

The pinned conservation tolerance is relative `2e-2`; the measured worst
case leaves approximately 1.12x headroom. The aggregate COM is inside the
mesh, 5.52 cm above the centered pelvis, and within 2.47 mm of direct
whole-mesh integration.

## Neutral per-part mass

| Native part | Mass (kg) |
|---|---:|
| `pelvis` | `15.470751762390137` |
| `left_hip` | `7.614247798919678` |
| `right_hip` | `7.620762348175049` |
| `spine1` | `6.0588555335998535` |
| `left_knee` | `3.113943576812744` |
| `right_knee` | `3.098294258117676` |
| `spine2` | `5.08026123046875` |
| `left_ankle` | `0.6279234290122986` |
| `right_ankle` | `0.6338295340538025` |
| `spine3` | `11.82639217376709` |
| `left_foot` | `0.1507422775030136` |
| `right_foot` | `0.15893858671188354` |
| `neck` | `0.5915396213531494` |
| `left_collar` | `0.6353169679641724` |
| `right_collar` | `0.6147966980934143` |
| `head` | `4.105005264282227` |
| `left_shoulder` | `1.6499942541122437` |
| `right_shoulder` | `1.7146587371826172` |
| `left_elbow` | `0.9918727278709412` |
| `right_elbow` | `1.0295754671096802` |
| `left_wrist` | `0.23914365470409393` |
| `right_wrist` | `0.2520706057548523` |
| `left_hand` | `0.07340041548013687` |
| `right_hand` | `0.07862172275781631` |

## SMPL-X and dynamics measurements

Neutral SMPL-X mass smoke values use the same density and vary slightly with
part reduction because centroid-cap partition error changes with ownership.

| `use_hands` | `use_face` | BetterRobot bodies | Total physical mass (kg) |
|---|---|---:|---:|
| `True` | `True` | 56 | `72.39342498779297` |
| `False` | `True` | 26 | `72.41643524169922` |
| `True` | `False` | 53 | `72.59387969970703` |
| `False` | `False` | 23 | `72.61688995361328` |

- Neutral SMPL gravity RNEA returned finite `(75,)` generalized force. CRBA
  was symmetric positive definite with minimum eigenvalue
  `3.2079296943265945e-05`.
- Two shape samples produced masses `[73.43093872070312,
  70.85686492919922]` kg, RNEA shape `(2, 75)`, CRBA shape `(2, 75, 75)`,
  and minimum CRBA eigenvalue `2.7201283955946565e-05`.
- Float32 `betas -> total mass` gradcheck passed in `0.2644321033731103` s.
- Float32 `betas -> RNEA tau` backward passed with finite nonzero gradients;
  the measured forward/backward run took `0.08441013097763062` s.
- The mass-enabled CUDA RNEA/CRBA smoke passed on pinned RTX 6000 Ada GPU 3
  in 4.20 s.

## Segmentation and closure findings

- Argmax differs from the shipped canonical membership for 113 of 6890
  vertices. This comparison treats any of the asset's overlapping boundary
  memberships as a match; 582 vertices belong to more than one canonical
  part and no vertex is uncovered.
- Majority assignment has an explicit all-distinct rule: the face's first
  vertex wins. Boundary traversal uses oriented half-edges, so pinched
  boundaries are separated into cycles rather than incorrectly rejected as
  branching. Every generated SMPL part and the synthetic split-box fixture
  had undirected edge incidence exactly two after capping.
- A warmed neutral-SMPL `with_shape` averaged
  `0.010191699117422104` s without mass and `0.041501314379274845` s with
  mass: 4.072070211367486x slower and 31.30961526185274 ms additional work.
  This is the measured evidence for the opt-out flag.

## Verification

- `uv lock --check`: 76 root-project packages, current.
- `uv lock --check --project tools/parity/smplx`: 38 oracle packages,
  current.
- `uv run pytest tests -q -rs`: **98 passed, 5 skipped, 7 warnings in
  91.50 s**. Skips are three sandbox CUDA tests and both explicit absent
  SMPL-X on-demand golden cases. Warnings are four required float32-gradcheck
  notices, PyTorch's CSR-beta notice, and two previously recorded SciPy
  pickle deprecations.
- Targeted CUDA mass/dynamics test: **1 passed in 4.20 s**.
- `uv run ruff check src tests`: passed.
- Independent review found no remaining cap orientation, link-frame,
  universe-index, batch, gradient, or SMPL-X partial-model blocker after the
  licensed-fixture CUDA skip was added.
- The combined `models/` and `references/` path/size/mtime manifest remained
  `1e12ec801fada88a15763f372770d3e065e4d1015a680a3f0e50917cb52e2258`.
- BetterRobot remained clean on `dev`; no dependency-side files were changed.

## Deviations and BetterRobot findings

- **Canonical segmentation deviation:** the planned preference for
  `models/star/smpl_3d_segmentation.npy` was rejected after inspection. Its
  part tables contain duplicate exterior triangles `(1819, 1818, 2881)` in
  parts 9/13 and `(5281, 6340, 5279)` in parts 9/14. Those duplicates create
  edge incidences one and three, so the tables are not closed two-manifold
  inputs. The generic argmax path is used instead; the asset was not modified.
- **BetterRobot validation deviation:** contrary to the plan's premise,
  `Inertia.from_mesh` does not validate closed edge incidence or mixed
  winding. A box missing two faces and a box with one reversed face were both
  accepted with mass `6.666666030883789`. BetterHuman therefore verifies
  boundary topology while constructing caps and asserts final edge incidence
  in tests. A general GPU-safe mesh validator belongs in BetterRobot and was
  not duplicated here; this order made no BetterRobot change.
- The optional mass flag is not a speculative deviation: the plan explicitly
  allowed it when cost was material, and the measured 4.07x shape-bake cost
  met that condition.
