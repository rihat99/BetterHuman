# Vision

BetterHuman is a PyTorch library of **parametric human body models on a
robotics core**. It unifies models that today live in incompatible,
single-model codebases — SMPL, SMPL-H, SMPL-X, MANO, STAR, SUPR, MHR, Anny,
and future ones — under one clean API, backed by
[BetterRobot](../../BetterRobot) for kinematics, dynamics, and optimization.

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
   BetterRobot kinematic tree (free-flyer root, spherical joints). FK, IK,
   Jacobians, dynamics, trajectory optimization — all inherited from
   BetterRobot, batched and differentiable. Every model provides
   `from_classic` / `to_classic` to translate its native convention
   losslessly, so existing datasets and codebases remain reachable.
3. **Model/Data split, human edition.** Following Pinocchio and BetterRobot:
   static topology in the model structure, differentiable model tensors in
   model values, per-query state in `Data`. Shape parameters (betas) live on
   the model side — changing shape updates the model once; pose inference
   afterwards is pure fast robotics FK. Joint angles live in `Data`.
4. **LEGO composability.** Partial models are constructor parameters, not
   separate classes (SMPL-X without expressions, without articulated
   fingers). Human-specific residuals plug into the BetterRobot optimizer.
   Conversion between body models and retargeting to humanoid robots are
   modules over the same core.
5. **Closer to robotics than to graphics.** Mass and inertia are first-class
   (dynamics on human bodies must work out of the box). Meshes and skinning
   are outputs, not the organizing principle.

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
nothing here — BetterRobot already loads URDF — so the humanoid side of
BetterHuman is limited to human↔robot tools (retargeting), with a namespace
reserved. During alpha, BetterHuman depends on a local editable checkout of
BetterRobot and the two evolve together; genuinely general capabilities
discovered here (e.g. gaps in spherical-joint support or mesh-based inertia)
are contributed to BetterRobot, not forked.
