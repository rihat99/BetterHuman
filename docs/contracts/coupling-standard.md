# Coupling standard

The shapes, layouts, identifiers and file formats through which BetterVideoReconstruction and BetterRetarget read
BetterHuman. The standard changes only by addition: removing or changing a line needs a decision record that shows
each consumer's migration (BetterRobot 0026). It names no function, so a function can be renamed without touching it.
Units are SI (metres, radians, kilograms, seconds). `B...` are the leading batch axes of an input, `I...` those of the
identity values.

BetterHuman's bodies are BetterRobot models, and BetterRobot's coupling standard,
`BetterRobot/docs/contracts/coupling-standard.md`, holds for them and for their states unchanged: the layouts of `q`,
`v`, poses, twists, wrenches, inertias and Euler angles, joint 0 and joint 1, joint order, mimic joints, identifiers
and batch axes. This page does not repeat it; it adds what a BetterHuman body fixes on top.

Every line of a `text` block below is part of the standard. `tests/contract/test_coupling_standard.py` checks each one
against the library, and holds the lines to their frozen copy `tests/contract/coupling_standard_lines.txt`: a line
added here is added there in the same commit, and a line there never leaves this page. The same test is BetterHuman's
coupling test against BetterRobot's page: it checks the BetterRobot lines BetterHuman relies on against the
BetterRobot it runs with.

## The body as a BetterRobot model

```text
root       joint 1 is a free flyer placed without rotation: q[..., :3] moves it along the world axes
           and q[..., 3:7] is its world rotation
up         y: gravity (0, -9.81, 0) in every family
native     J native joints in the family's own order; joint_ids (J,) holds the BetterRobot joint of each
           joint joint_ids[j] moves the body named after native joint j; a native world anchor is joint 0, the universe
```

`J` is the family's number of native (skinning) joints and `V` its number of vertices. MHR's and SOMA's native joint 0
is a world anchor; the SMPL family's native joint 0 is its root, joint 1.

## HumanState

```text
batch                    a state's batch is broadcast(B..., I...): B... of q, I... of the identity values
joint_transforms         (..., J, 3, 4)     world [R | p] of the native joints, in native order
joint_poses              (..., J, 7)        the same placements as poses [tx, ty, tz, qx, qy, qz, qw]
joint_translations       (..., J, 3)        their world positions: BetterRobot's joint translations at joint_ids
rest_joint_translations  (..., J, 3)        the native joints in the skinning rest frame
skin_affines             (..., J, 3, 4)     per native joint, the affine (it may scale) from the rest frame to the world;
                                            it carries rest_joint_translations to joint_translations
vertices                 (..., V, 3)        the posed mesh, in the family's vertex order
frame_poses              (..., nframes, 7)  world poses of the model's frames in frame order, contact frames included
```

A transform `[R | p]` is rigid; an affine may scale (BetterRobot 0027). The batch rule is BetterRobot's (BetterRobot
0028).

## Files

```text
format                            direction    owner         promise
better_human.mhr                  read         BetterHuman   format and format_version fields; every version written stays readable
better_human.soma                 read         BetterHuman   format and format_version fields; every version written stays readable
better_human.frame_set            read, write  BetterHuman   format and format_version fields; every version written stays readable
better_human.segmentation         read         BetterHuman   format and format_version fields; every version written stays readable
better_human.density_tables       read         BetterHuman   format and format_version fields; every version written stays readable
better_human.studio.segmentation  read, write  BetterHuman   format and format_version fields; every version written stays readable
better_human.studio.pose          read, write  BetterHuman   format and format_version fields; every version written stays readable
```

These are all seven formats BetterHuman versions. BetterRobot's page lists the first three, the ones BetterRobot 0030
names. The converted archives are record [0012](decisions/0012-bh-converted-archives.md); the frame set, the package
data and the studio formats are record [0013](decisions/0013-bh-package-data-and-load-speed.md). A format this page
does not list carries no promise. `tests/contract/test_formats.py` checks each one's promise, and each one's spec is a
page under `docs/reference/formats/`.
