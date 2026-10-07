# Contracts

BetterHuman follows BetterRobot's design contracts. They live in the BetterRobot repository under
`BetterRobot/docs/contracts/` (start at `BetterRobot/docs/contracts/00-index.md`); rule IDs (`C-*`) are cited
here as written there. A cross-repository link is not possible, so cite the page path and the rule ID.

BetterHuman adds its own rules (`H-*`), its numeric conventions, weight-file layout, parity gates, baselines
and design principles in [`bh-addendum.md`](bh-addendum.md).

## BetterRobot contract families that apply unchanged

| BR page | rule IDs | how it applies in BetterHuman |
|---|---|---|
| `BetterRobot/docs/contracts/02-tensors.md` | `C-TENS-*` | trailing event axes, arbitrary leading batch axes; preserve dtype and device; no silent cast, move or squeeze; no host sync in hot paths |
| `BetterRobot/docs/contracts/03-state-mutability.md` | `C-STATE-*` | public ops never mutate inputs; `HumanState` is created per query and never mutated (it only memoises `vertices()`) |
| `BetterRobot/docs/contracts/07-dependencies-layering.md` | `C-DEP-*` | imports point down the package DAG; optional dependencies stay lazy; BH uses only BR's public surface |
| BR numerical contracts (pose `[t, qxyz, qw]`, twist `[linear, angular]`, `nq != nv`, `Model.integrate`) | `C-*` | `q` is a BR configuration; move on the manifold with `identity.model.integrate` |
| BR naming, style, testing and packaging pages | `C-*` | same conventions; BH-specific deviations are listed as `H-*` rules |

When a BetterHuman rule and a BetterRobot rule disagree, BetterRobot's numerical contracts win.

## Enforcement in BetterHuman

| test | rules |
|---|---|
| `tests/contract/test_br_imports.py` | `C-DEP-*`: only BetterRobot's public surface is imported |
| `tests/contract/test_layer_dependencies.py` | `H-LAYER-1`: BetterHuman package DAG |
| `tests/contract/test_keyword_only.py` | `H-API-2` / `C-API-2`: keyword-only arguments, with an allowlist of current violations |
| `tests/parity/` | `H-PARITY-1`: parity gates |
