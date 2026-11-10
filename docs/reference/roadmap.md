# Roadmap

What BetterHuman plans to build. What it will not build is in [Scope](../contracts/scope.md); what landed is in the
[changelog](changelog.md).

The plan below lists one *brick* (a self-contained piece of work) per row,
why it is wanted, what must exist first, and when it is planned:

- **now**: landed in this version;
- **next**: the following version; the design is settled;
- **later**: wanted, but waits for its prerequisite or a real caller;
- **maybe**: kept open; it starts only if a concrete need appears.

No brick starts without an owner decision, and the API is not frozen during
alpha. Robot-side work (optimizers, dynamics, collision, viewer) is planned in
BetterRobot's roadmap.

## Families and bodies

| Brick | Why | Prerequisite | When |
|---|---|---|---|
| STAR family | sparse, shape-aware correctives | `identity` in `corrective_features` (now) | next |
| Rig-agnostic contact frames | ground contact, dynamics | BetterRobot contact Jacobians | next |
| Family transfer and a shared identity protocol | moving a person between families | STAR | later |
| Anny family; body measurements; an asset manifest | more bodies, dataset tooling | identity protocol | later |
| Biomechanical joints (coupled knee, scapula) as BetterHuman extensions | anatomical motion | a family that needs them | later |
| Vertex frames and the camera moved to BetterRobot; garments | apparel | a second caller in BetterRobot | maybe |

## Motion and data

| Brick | Why | Prerequisite | When |
|---|---|---|---|
| `Motion` type for clips | one clip object for import, studio and dynamics | none; BetterRobot has no clip or trajectory type to build on | next |
| Mocap import: C3D, BVH, AMASS | gait labs, datasets | `Motion` type | next |
| Marker sets and virtual markers for OpenSim-style marker IK | biomechanics labs | mocap import | later |
| Force-plate / ground-reaction input | gait dynamics | BetterRobot `external_forces` input, contact frames | later |
| Muscles and musculotendon models | biomechanics | marker sets | maybe |
