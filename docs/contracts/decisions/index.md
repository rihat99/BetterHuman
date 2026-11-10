# Decision records

A decision record states one decision that shapes BetterHuman: its context, the decision, an example, the
consequences, the alternatives, the compatibility cost and what enforces it. An accepted record is never edited
except its Status line; a change is a new record that supersedes it. File name `NNNN-kebab-title.md`.

| Record | Title | Status |
|---|---|---|
| [0001](0001-human-model.md) | BetterHuman's `Body` becomes a frozen `HumanModel` built by one factory | Accepted |
| [0002](0002-rig-abstract-base.md) | `Rig` is an abstract base with declared hooks | Accepted |
| [0003](0003-one-skinning-path.md) | Every rig skins through inverse-bind affines | Accepted |
| [0004](0004-classic-api.md) | One classic API: `bh.from_classic` and `bh.to_classic` | Accepted |
| [0005](0005-dtype-and-device-at-load.md) | BetterHuman chooses dtype and device at load; BetterRobot keeps `Model.to` | Accepted |
| [0006](0006-level-one-loader.md) | `bh.load` has an explicit keyword-only signature | Accepted |
| [0007](0007-mass-and-density.md) | One `density` parameter with the shipped table as default | Accepted |
| [0008](0008-typed-family-spec.md) | `FamilySpec` has typed fields and no `meta` | Accepted |
| [0009](0009-studio-placement.md) | The studio is `better_human.studio`, an optional application subpackage | Accepted |
| [0010](0010-human-viz-surface.md) | BetterHuman's `viz` is one `layers` module with state-driven functions | Accepted |
| [0011](0011-human-package-layout.md) | BetterHuman packages: `io, mass, core, families, terms, viz, studio`, with two edge tables | Accepted |
| [0012](0012-bh-converted-archives.md) | BetterHuman archives carry `format`, `format_version` 2 and `provenance` | Accepted |
| [0013](0013-bh-package-data-and-load-speed.md) | BetterHuman package data is versioned and loads without pickle | Accepted |
| [0014](0014-golden-provenance.md) | Every golden file stamps its provenance | Accepted |
| [0015](0015-licensed-goldens-out-of-git.md) | Licensed-derived goldens live in the models directory, not in git | Accepted |
| [0016](0016-random-weight-goldens.md) | Random-weight goldens run the SMPL-family parity machinery without licensed weights | Accepted |
| [0017](0017-models-directory.md) | One way to find the weights: `io.models_dir()` | Accepted |
| [0018](0018-bh-apache-licence.md) | BetterHuman is licensed Apache-2.0 | Accepted |
| [0019](0019-viz-imports-core-only.md) | `viz` imports `core` only: the `viz -> io` edge goes | Proposed |

## Records that live in BetterRobot

Decisions that bind both libraries are recorded once, in BetterRobot, and cited here as `BetterRobot NNNN`.
A cross-repository link is not possible, so this table gives each one's path in the BetterRobot repository.
Records that bind BetterRobot alone sit in the same folder.

| Record | Title |
|---|---|
| `BetterRobot/docs/contracts/decisions/0010-equality-and-hashing.md` | Identity equality for models and states, one by-value static key |
| `BetterRobot/docs/contracts/decisions/0023-naming-sweep.md` | Naming sweep: full words, plural axes, `_ids` for integer indices |
| `BetterRobot/docs/contracts/decisions/0026-coupling-standard-page.md` | One coupling-standard page in BetterRobot, tested by every consumer |
| `BetterRobot/docs/contracts/decisions/0027-shared-vocabulary.md` | One glossary for `KinematicState` and `HumanState` |
| `BetterRobot/docs/contracts/decisions/0028-batch-axes.md` | States broadcast `B...` against model values `I...`; `optim` keeps `(B, *event)` |
| `BetterRobot/docs/contracts/decisions/0029-frames-and-identifiers.md` | Names at every user boundary, ids inside |
| `BetterRobot/docs/contracts/decisions/0030-formats-in-the-standard.md` | The formats that cross repositories, listed in the standard |
| `BetterRobot/docs/contracts/decisions/0031-error-vocabulary-across-the-stack.md` | BetterHuman raises BetterRobot's errors and roots its own under them |
| `BetterRobot/docs/contracts/decisions/0033-test-tiers.md` | Three test tiers, each one local command; a missing oracle fails |
| `BetterRobot/docs/contracts/decisions/0034-python-312.md` | Python 3.12 is the floor and the development version |
| `BetterRobot/docs/contracts/decisions/0035-shared-testing-module.md` | Shared test and docs code lives once, in `better_robot.testing` |
