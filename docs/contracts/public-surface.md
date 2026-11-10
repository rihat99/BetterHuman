# Public surface

A name is public when every segment of its dotted path is listed in its parent's `__all__`; everything else may
change without notice. This page is the one definition of what BetterHuman promises: what users may do, what the
maintainers keep in return, and which changes count as compatible. It follows BetterRobot's page,
`BetterRobot/docs/contracts/public-surface.md`, which a BetterRobot name keeps inside BetterHuman.

## Both sides

| | |
|---|---|
| Public | a dotted path whose every segment is in its parent's `__all__`; the root lists every subpackage (`core`, `families`, `io`, `mass`, `terms`, `viz`, `studio`) and the first-level names `load`, `FamilySpec`, `HumanModel`, `Identity`, `HumanState`, `from_classic`, `to_classic` and `__version__`; the tracked snapshot `tests/contract/public_api.json` lists every public path with its signature |
| Canonical | the root for a first-level name (record [0021](decisions/0021-root-imports-every-subpackage.md)), else the package whose `__all__` lists the name |
| Users may | call public names; read a body's BetterRobot `Model` and `KinematicState` with BetterRobot's own functions; write an optimisation term as a plain function over a `HumanState`; draw a body with BetterRobot's viewer and its `Layer` subclasses |
| Users may not | subclass a concrete class (every one is `@final`) or the abstract `Rig` and `ClassicFormat`, whose subclasses live in-tree with their families; rely on the order of a dataclass's fields (build a record by keyword: the keyword-only rule covers functions and methods, not dataclass fields); import a `_` name, or `tests/`, `examples/` or `tools/` |
| Experimental | nothing in this release |

## Horizons

- Python API before 1.0: a minor release may change or remove a public name, with a migration line in the
  changelog and a row on the migration page; there is no deprecated alias.
- Files: every file format BetterHuman reads or writes keeps reading every version it ever wrote (rule
  [H-FORMAT-1](rules.md#h-format-1-every-betterhuman-file-names-its-format-and-version-must)), with one exception:
  the studio pose format refuses version 0, the files the studio wrote before poses named a format, because they
  lack the joint names a version-1 file needs and no migration can recover them from the file alone
  ([studio pose](../reference/formats/studio_pose.md)).
- Numerical results: within the tolerances of each family's parity tests against its authors' code (rule
  [H-PARITY-1](rules.md#h-parity-1-every-family-matches-its-authors-code-within-the-parity-tolerances-must)).
- [Coupling standard](coupling-standard.md) (the shapes, layouts and file formats other repositories read):
  additions only (BetterRobot record 0026).

## Compatible changes

Every change not in a compatible row is breaking.

| Change | Class |
|---|---|
| Add a function, class, module or family | compatible |
| Add a keyword-only parameter with a default | compatible |
| Add a field to a frozen dataclass, keyword-only with a default | compatible |
| Add a version to a file format, with its migration | compatible |
| Add, remove or reorder a positional parameter | breaking |
| Change a default value (a loader's gender, number of coefficients, level of detail or density table) | breaking (behaviour) |
| Remove or rename a public name, field, parameter, identity group or stream group | breaking |
| Change a result beyond its parity tolerance | breaking |

## Checks

`tests/contract/test_api_snapshot.py` fails on any change of a public name or signature until the snapshot changes
in the same commit, which makes the class of each change visible in review; `tests/contract/test_changelog.py`
keeps every removed name on the migration page; `tests/contract/test_abstract_or_final.py` keeps every class
abstract or `@final`; `tests/contract/test_keyword_only.py` keeps the parameters after the second keyword-only in every function,
constructor (dataclass fields included), method, class method and static method of the `__all__` tree, a `*args`
counting as three; the public records' positional fields and `terms.baked_identity`'s `*groups` are today's counted
sites in `tests/contract/ratchets/positional_parameters.json`.
