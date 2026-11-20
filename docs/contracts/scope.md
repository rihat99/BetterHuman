# Scope

BetterHuman loads parametric human body models (SMPL, SMPL-H, SMPL-X, MANO, FLAME, MHR and SOMA) as batched,
differentiable BetterRobot models, with their skinning, pose correctives, part masses, contact frames, optimisation
terms and viewer layers. A feature outside this page is the owner's decision, and the page changes first.

## In scope

- Family loaders that read each family's weight files into one typed `FamilySpec`, and one `HumanModel` that bakes
  a subject and poses it.
- One rig base for the family-specific joints and binds, and one skinning path for every rig.
- Each family's own parameters, converted exactly at the boundary.
- Masses and inertias of the body's parts from its closed mesh.
- Cost terms over a posed body for BetterRobot's optimisers, viewer layers, and HumanStudio, the optional
  application for authoring parts, frames and poses.
- Converters from the authors' releases to the files the loaders read, and the versioned formats of those files.

## Non-goals

- Robotics: forward kinematics, Jacobians, dynamics, Lie algebra, solvers and the viewer come from BetterRobot; a
  capability that is not human-specific goes there.
- Rendering: BetterHuman produces geometry; images come from BetterRobot's viewer.
- Weights: BetterHuman never downloads, bundles or caches licensed model files; the user points it at them.
- Training body models: BetterHuman uses trained models; it does not train them.
- Fitting bodies to images, scans or markers: BetterRobot's optimisers and BetterHuman's terms are the building
  blocks of a fitter, which lives elsewhere.
- Retargeting between humans and robots: a separate package built on both libraries.
- Families from outside the repository: every family is a module under `better_human/families/`; there is no
  `register_family` hook.

## Inclusion bar

A family or feature lands when it:

1. is checked against its authors' released code, run in its own pinned environment, by a parity test that names
   that oracle (`tests/parity/`), and the test passes before the family is called supported;
2. goes through the one `FamilySpec`, `Rig` and skinning path, with its family-specific behaviour in its rig or its
   loader, never as a copy of another family's forward pass or a branch in `HumanModel`;
3. keeps every family's own parameter convention at the boundary, through its classic format, with an exact round
   trip;
4. is human-specific; otherwise it is a BetterRobot contribution.

## Commonly rejected

Reopening one of these needs new evidence, not preference.

| Idea | Reason |
|---|---|
| A class per family or per partial model (`SMPLX`, `SMPLXNoHands`) | copied forward passes, the `smplx` package's maintenance cost |
| Each family's native pose format inside the library | one pose convention, BetterRobot's `q`, everywhere inside |
| `to(dtype, device)` on BetterHuman objects | a second way to choose what the loader chose |
| A free-form `meta` mapping on the spec | untyped data that every rig reads differently |
| Downloading or shipping weights | the weights' licences are the user's to accept |
| Pickle files in the package | arbitrary code on load, slow reads |
