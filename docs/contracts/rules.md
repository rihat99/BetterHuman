# BetterHuman rules

The rules BetterHuman adds to BetterRobot's contracts, in BetterRobot's format: an ID (`H-<AREA>-<n>`) and a strength
in the heading, the rule, its rationale, an example and its enforcer, a test that fails when the rule is broken.
Every rule here is a must: it is never violated. The [index](index.md) lists them with the BetterRobot rules that
hold here unchanged. An ID is never reused: `H-API-2` (keyword-only arguments) is retired, because it was
BetterRobot's `C-API-2` under a second name, which BetterHuman now cites directly.

## H-LAYER-1 Imports follow the edge tables (MUST)

Each unit of `better_human` imports only the units of its row in `ALLOWED`, and only the BetterRobot packages of
its row in `BR_ALLOWED` (`tests/contract/test_layer_dependencies.py`; decision record
[0011](decisions/0011-human-package-layout.md) and the records that amend it). The [architecture
page](../concepts/architecture.md#the-package-map) shows both tables. No module imports a BetterHuman or BetterRobot
module by a computed name (`importlib.import_module`, `__import__`), which the import walker cannot see.

Rationale: the body model (`core`) and the file readers (`io`) stay usable without any family table or viewer, and
each BetterRobot package a unit uses is a decision, not an accident.
Example: `src/better_human/terms/` imports `core` and no other unit; `mass` imports no unit and reads no file.

Enforcement: `tests/contract/test_layer_dependencies.py`.

## H-LAYER-2 BetterRobot is imported through its public tree (MUST)

Every `better_robot` path a module imports is a path of BetterRobot's `__all__` tree, and a first-level name of
BetterRobot's root comes from the root: `from better_robot import Viewer`, never
`from better_robot.viewer import Viewer`. The one exception: the studio and `examples/03_ik_reach.py` import `better_robot.viewer.tools` (`PoseGizmo`,
`CameraFrustum`, the ray and selection helpers), which is outside BetterRobot's `__all__` tree; BetterRobot's owner decides whether that
module becomes public (an open question of record [0021](decisions/0021-root-imports-every-subpackage.md)).

Rationale: BetterRobot's compatibility promise covers its public tree only, and one import path per name keeps a
rename in BetterRobot to one edit here.
Example: `src/better_human/studio/app.py` imports `Viewer` from `better_robot`.

Enforcement: `tests/contract/test_br_imports.py`.

## H-API-1 One body API across families (MUST)

A family is a loader function in `better_human.families` that returns a `FamilySpec`, and `bh.load` dispatches to
every such loader; what a family does differently lives in its `Rig` and its loader. A partial model is a loader
option (`hands=False`) or `families.drop_joints`, never a subclass: no class derives from a `@final` class such as
`HumanModel`, `Identity` or `HumanState` (decision records [0001](decisions/0001-human-model.md),
[0002](decisions/0002-rig-abstract-base.md), [0006](decisions/0006-level-one-loader.md)).

Rationale: one forward pass to test and to keep in parity, instead of one copy per family.
Example: `src/better_human/families/smpl_family.py` serves all five SMPL-family loaders from one table each.

Enforcement: `tests/contract/test_keyword_only.py::test_load_shows_every_option_and_dispatches_to_every_family_loader`,
`tests/contract/test_abstract_or_final.py`.

## H-API-3 Dtype and device are chosen once, at load (MUST)

A family loader and `bh.load` take `dtype` (`torch.float32` or `torch.float64`) and `device`, and every tensor of the
body is built there; no BetterHuman class or function is named `to`, and a tensor in another dtype than the model's
raises instead of being cast (decision record [0005](decisions/0005-dtype-and-device-at-load.md)).

Rationale: one place chooses the precision; a silent cast hides a caller's mistake.
Example: `bh.load("smpl", dtype=torch.float64)`; `human.bake(betas=torch.zeros(10, dtype=torch.float64))` on a
float32 body raises `DtypeMismatchError`.

Enforcement: `tests/contract/test_dtype_device.py`.

## H-ASSET-1 Licensed weights and goldens computed from them are never tracked (MUST)

Git holds no `.npz` file but the shipped package data, the random-weight archives and their goldens; the licensed
weights and the goldens computed from them live in `$BETTERHUMAN_MODELS_DIR` (decision records
[0015](decisions/0015-licensed-goldens-out-of-git.md), [0016](decisions/0016-random-weight-goldens.md)).

Rationale: the weights' licences are the user's to accept; a tracked copy would redistribute them.
Example: the SMPL-X parity goldens live in `$BETTERHUMAN_MODELS_DIR/goldens/smplx_neutral/`; the random-weight ones
in `tests/parity/smplx/random/smplx/`.

Enforcement: `tests/contract/test_golden_provenance.py::test_git_holds_no_npz_but_the_allowed_ones`.

## H-ASSET-2 Weights resolve from an explicit path, else from `io.models_dir()` (MUST)

An explicit `path` argument wins; otherwise the file is `io.weight_path(folder, filename)` under `io.models_dir()`,
the one reader of `$BETTERHUMAN_MODELS_DIR`, which raises with the fix when the variable is unset or names no folder.
A missing file raises naming the path; nothing guesses a default, and no test, example, tool or benchmark builds a
`models` path of its own (decision record [0017](decisions/0017-models-directory.md)).

Rationale: one lookup rule, so a missing file has one fix.
Example: `src/better_human/io/load.py::weight_path`.

Enforcement: `tests/io/test_load.py`, `tests/contract/test_models_dir.py`.

## H-FORMAT-1 Every BetterHuman file names its format and version (MUST)

Each file format BetterHuman reads or writes is in `io.formats.LATEST` with a spec page under
`docs/reference/formats/`; a file carries `format` and `format_version`, a past version reads through its
migration, and a later version or another format is refused naming both (decision records
[0012](decisions/0012-bh-converted-archives.md), [0013](decisions/0013-bh-package-data-and-load-speed.md)). No module
loads a pickle.

Rationale: a file outlives the code that wrote it; the reader, not the user, carries old files forward.
Example: `io.read_frame_set(path)` reads a version-1 frame set and refuses a version-2 one naming the newest it
knows.

Enforcement: `tests/contract/test_formats.py`.

## H-PARITY-1 Every family matches its authors' code within the parity tolerances (MUST)

Each family's forward pass, and its bake where the oracle exposes one, agrees with the authors' released code, run
in its own pinned environment, within the tolerances the parity tests hold; a tolerance never moves to land a
change ([Test suite](../reference/tests.md#parity)).

Rationale: parity with the original is the safety net that every refactor runs against.
Example: SMPL-X vertices and joints within 1e-6 m of the `smplx` package in float32 and float64.

Enforcement: `tests/parity/smplx/test_smpl_parity.py`, `tests/parity/smplx/test_smplx_parity.py`,
`tests/parity/smplx/test_random_weights.py`, `tests/parity/mhr/test_mhr_parity.py`,
`tests/parity/mhr/test_mhr_torchscript.py`, `tests/parity/soma/test_soma_parity.py`.

## H-PARITY-2 Every golden stamps its provenance (MUST)

A golden file carries a `provenance` object naming its oracle, its generator, the hash of the weights it was
computed from, its dtype and its seed; a tracked golden names a random-weight archive (decision record
[0014](decisions/0014-golden-provenance.md)).

Rationale: a golden that cannot say what produced it cannot be regenerated or trusted.
Example: `tests/parity/smplx/random/smpl/` holds goldens whose provenance names `smpl_random.npz`.

Enforcement: `tests/contract/test_golden_provenance.py`.

## H-STUDIO-1 The studio is an application on the public surfaces (MUST)

`better_human.studio` imports only names of BetterHuman's and BetterRobot's `__all__` trees from outside itself,
reads no `_` attribute its own class does not define (a studio layer may call the subclass hooks BetterRobot's
`Layer` documents), and writes no environment variable (decision record [0009](decisions/0009-studio-placement.md)).
A viewer feature that is not human-specific goes to BetterRobot. The one exception is H-LAYER-2's:
`better_robot.viewer.tools`, outside BetterRobot's `__all__` tree, until BetterRobot's owner decides whether it
becomes public.

Rationale: the studio is the first application on the libraries; whatever it needs, every application needs.
Example: `src/better_human/studio/layers.py::SegmentationLayer` subclasses `better_robot.viewer.Layer`.

Enforcement: `tests/contract/test_studio_boundary.py`.

## H-STUDIO-2 Studio data lives under `<data root>/{segmentations,frames,poses}/<model key>/` (MUST)

Everything the studio saves is a versioned JSON file under the data root the user passes: vertex segmentations,
frame sets and poses each in their own folder, keyed by the model, so a set saved for one body never loads onto
another. A name typed in the studio is one file name; the studio refuses any other and writes nothing.

Rationale: scripts read the same files without the app.
Example: `src/better_human/studio/persistence.py::StudioPaths`.

Enforcement: `tests/studio/test_persistence.py::test_studio_paths_layout`,
`tests/studio/test_app.py::test_the_studio_runs_on_the_stand_in_server`,
`tests/studio/test_app.py::test_a_save_name_that_is_not_one_file_name_is_refused_and_writes_nothing`.

## H-LICENSE-1 One licence, Apache-2.0 (MUST)

The `LICENSE` text, the `pyproject.toml` licence and the README's licence line all say Apache-2.0, neither the
README nor `pyproject.toml` names another code licence, and the README keeps that body-model weights and data derived
from them have their own terms (decision record [0018](decisions/0018-bh-apache-licence.md)).

Rationale: the code's licence is stated once and agrees everywhere it is stated.
Example: `pyproject.toml` declares `license = "Apache-2.0"`.

Enforcement: `tests/contract/test_license.py`.
