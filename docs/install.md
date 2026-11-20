# Install and weights

BetterHuman needs Python 3.12 or newer, PyTorch and BetterRobot. The project uses [uv](https://docs.astral.sh/uv/).
While both libraries are in development, BetterHuman reads BetterRobot from a checkout beside its own, so clone both
into one folder:

```bash
git clone https://github.com/rihat99/BetterRobot
git clone https://github.com/rihat99/BetterHuman && cd BetterHuman
uv sync --extra studio
```

This installs the package, the tools for tests and docs, and, with the `studio` extra, the browser viewer and the
studio. On Linux `uv sync` installs PyTorch's CUDA 12.9 build; the device of your tensors decides where a call runs.

To check the install without any weights, run the fast test tier (about 15 seconds):

```bash
uv run pytest tests/contract -q -m "not slow and not cuda and not licensed"
```

## The weights

BetterHuman ships no body weights: their licences are yours to accept. Download SMPL, SMPL-H, SMPL-X, MANO and FLAME
from their sites ([SMPL](https://smpl.is.tue.mpg.de/), [SMPL-X](https://smpl-x.is.tue.mpg.de/) and their
siblings), and MHR and SOMA from their authors' releases. Put the files in one folder, in this layout, and name it in
`$BETTERHUMAN_MODELS_DIR`:

| Family | Folder | File |
|--------|--------|------|
| SMPL | `smpl` | `SMPL_NEUTRAL.npz`, `SMPL_MALE.npz`, `SMPL_FEMALE.npz` |
| SMPL-H | `smplh` | `smplh/{neutral,male,female}/model.npz` |
| SMPL-X | `smplx` | `SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz` |
| MANO | `mano` | `MANO_RIGHT.npz`, `MANO_LEFT.npz` |
| FLAME | `flame` | `generic_model.npz`, `male_model.npz`, `female_model.npz` |
| MHR | `MHR` | `converted/mhr_lod{0..6}.npz` (level of detail 1 by default) |
| SOMA | `SOMA` | `soma_mid.npz` |

```bash
export BETTERHUMAN_MODELS_DIR=/path/to/models
```

Three releases need converting first. Each converter runs from the repository root and writes into the folder above:

- SMPL comes as a Chumpy pickle. `uv run tools/convert/smpl/convert_chumpy.py <pickle> "$BETTERHUMAN_MODELS_DIR/smpl/SMPL_NEUTRAL.npz" --model smpl`
  reads it without Chumpy; run it once per gender.
- MHR comes as its authors' asset folder. `tools/convert/mhr/convert_mhr.py --assets-dir ... --output-dir ...`
  writes one archive per level of detail; it needs `pymomentum`, from the conda environment in
  `tools/parity/mhr/environment.yml`.
- SOMA comes as the SOMA-X v0.3.4 release. `tools/convert/soma/convert_soma.py` writes `soma_mid.npz`;
  `tools/convert/soma/README.md` says what it needs.

## Check the setup

`bh.load(family)` finds the family's default file under `$BETTERHUMAN_MODELS_DIR`:

```python
import better_human as bh

for family in ("smpl", "smplx", "mhr", "soma"):
    human = bh.load(family)
    joints, vertices = len(human.spec.joint_names), human.spec.num_vertices
    print(f"{family:5} nq={human.model.nq} joints={joints} vertices={vertices}")
```
```output
smpl  nq=99 joints=24 vertices=6890
smplx nq=223 joints=55 vertices=10475
mhr   nq=132 joints=127 vertices=18439
soma  nq=311 joints=110 vertices=18056
```

An unset variable or a missing file raises `AssetNotFoundError` naming what to set. A file outside the layout is
passed as the path, `bh.load("smpl", "/data/SMPL_MALE.npz")`, and the path wins over the folder.

Next, read [Families and files](families.md) for each family's options.
