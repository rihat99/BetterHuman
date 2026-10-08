# Vision

BetterHuman is a PyTorch library of **parametric human body models on a
robotics core**. It unifies models that today live in incompatible,
single-model codebases — SMPL, SMPL-H, SMPL-X, MANO, STAR, SUPR, MHR, Anny,
and future ones — under one clean API, backed by
BetterRobot for kinematics, dynamics, and optimization.

## The problem

Every parametric body model ships its own repository with its own
conventions: its own forward pass, its own pose format, its own file layout,
its own quirks (SMPL's root translation is not centered at the root joint;
MHR has its own rig conventions; chumpy-era weight files need a museum-grade
NumPy to open). Switching models means rewriting your pipeline. None of them
speak robotics: no configuration manifold, no Jacobians, no dynamics, no
solver — just a graphics-style forward function from parameters to vertices.

## The idea

1. **One API, many models.** An abstract parametric-body class defines the
   contract; each model is a thin subclass over shared machinery (LBS,
   blend shapes, correctives, joint regression). Users switch body models by
   changing one string, not their pipeline.
2. **Robotics-native state.** Pose is a configuration vector `q` on a real
   BetterRobot kinematic tree: spherical joints for the SMPL family, and
   named revolute and mimic joints for MHR. FK, IK, Jacobians, dynamics, and
   trajectory optimization come from BetterRobot, batched and differentiable.
   Every family provides
   `from_classic` / `to_classic` to translate its native convention
   losslessly, so existing datasets and codebases remain reachable.
3. **Model/Data split, human edition.** Following Pinocchio and BetterRobot:
   static topology in the model structure, differentiable model tensors in
   model values, per-query state in `HumanState`. Named shape groups live on the
   model side — changing shape updates the model once; vertex queries then run
   FK, pose correctives, and skinning without recomputing shape. Joint angles
   live in `q`, passed per query.
4. **LEGO composability.** Partial models are constructor parameters, not
   separate classes (SMPL-X without expressions, without articulated
   fingers). Human-specific residuals plug into the BetterRobot optimizer.
   Conversion between body models is a module over the same core; retargeting
   to humanoid robots is a separate package built on BetterHuman and BetterRobot.
5. **Closer to robotics than to graphics.** Mass and inertia are first-class.
   Models with independent coordinates support solve-based dynamics directly;
   MHR's current overcomplete coordinates expose a documented singular CRBA.
   Meshes and skinning are outputs, not the organizing principle.

## Non-goals

- **No rendering pipeline.** Visualization goes through BetterRobot's viewer;
  BetterHuman produces geometry, not images.
- **No robotics reimplementation.** If it is not human-specific, it belongs
  in BetterRobot. BetterHuman contains no FK, no solvers, no Lie math.
- **No weight redistribution.** Most model weights are license-gated; users
  download them and point BetterHuman at the files. We ship converters, not
  weights.
- **No training frameworks.** BetterHuman consumes trained body models; it
  does not train them.

## Relationship to BetterRobot

BetterHuman sits above BetterRobot the way `tasks` sits above `optim`: it
composes the core instead of copying it. Humanoid *robots* need almost
nothing here — BetterRobot already loads URDF — and human↔robot
retargeting is a separate package built on both, not part of BetterHuman.
During alpha, BetterHuman depends on a local editable checkout of
BetterRobot and the two evolve together; genuinely general capabilities
discovered here (e.g. gaps in spherical-joint support or mesh-based inertia)
are contributed to BetterRobot, not forked.

## Roadmap

The plan below lists one *brick* (a self-contained piece of work) per row,
why it is wanted, what must exist first, and when it is planned:

- **now**: landed in this version (listed below);
- **next**: the following version; the design is settled;
- **later**: wanted, but waits for its prerequisite or a real caller;
- **maybe**: kept open; it starts only if a concrete need appears.

No brick starts without an owner decision, and the API is not frozen during
alpha. Robot-side work (optimizers, dynamics, collision, viewer) is planned in
BetterRobot's roadmap.

### Landed in this version (now)

- `Rig.corrective_features(spec, model, q, *, identity=None)`: rigs can read
  shape coefficients, and the feature memo is invalidated when identity
  changes. This is breaking for third-party rigs
  ([addendum](../contracts/bh-addendum.md)).
- `HumanState.vertices` and `HumanState.frame_poses` are cached properties,
  like `joints`.
- `FamilySpec` documents every `meta` key that core reads, and a test keeps the
  list in sync.
- Time semantics: a clip is `(B, T, nq)` with `fps`/`times` following
  BetterRobot's viewer time model ([addendum](../contracts/bh-addendum.md)).
- The [studio](../studio.md) registers pointer tools through BetterRobot's
  `Viewer.on_pointer` / `clear_pointer` and picking helper, resets the active
  frame on a model switch, and caches the MHR tree per device.

### Families and bodies

| Brick | Why | Prerequisite | When |
|---|---|---|---|
| STAR family | sparse, shape-aware correctives | `identity` in `corrective_features` (now) | next |
| Rig-agnostic contact frames | ground contact, dynamics | BetterRobot contact Jacobians | next |
| Family transfer and a shared identity protocol | moving a person between families | STAR | later |
| Anny family; body measurements; an asset manifest | more bodies, dataset tooling | identity protocol | later |
| Biomechanical joints (coupled knee, scapula) as BetterHuman extensions | anatomical motion | a family that needs them | later |
| Vertex frames and the camera moved to BetterRobot; garments | apparel | a second caller in BetterRobot | maybe |

### Motion and data

| Brick | Why | Prerequisite | When |
|---|---|---|---|
| `Motion` type for clips | one clip object for import, studio and dynamics | BetterRobot `Trajectory` (now) | next |
| Mocap import: C3D, BVH, AMASS | gait labs, datasets | `Motion` type | next |
| Marker sets and virtual markers for OpenSim-style marker IK | biomechanics labs | mocap import | later |
| Force-plate / ground-reaction input | gait dynamics | BetterRobot `fext` input, contact frames | later |
| Muscles and musculotendon models | biomechanics | marker sets | maybe |

### Not on the roadmap

- **Fitting** (bodies to images, scans or markers) is not BetterHuman's job;
  BetterRobot's optimizer and BetterHuman's terms are the building blocks.
- **Retargeting** between humans and robots will be a separate package, not
  part of BetterHuman.
- **Third-party families** live in this repository during alpha;
  `register_family` is not offered.
- **Video reconstruction** sync and panels come later, through BetterRobot's
  viewer.
