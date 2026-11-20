# Load a family

This guide points BetterHuman at your licensed weights and loads each family with its options. BetterHuman never
downloads or ships weights: register at
the [SMPL](https://smpl.is.tue.mpg.de/) or [SMPL-X](https://smpl-x.is.tue.mpg.de/) site and download the files
there, and take MHR and SOMA from their authors' releases. [Convert weights](convert-weights.md) turns the files
that need it into the archives below.

## Point BetterHuman at the weights

Put the files in one folder, in this layout, and name it in `$BETTERHUMAN_MODELS_DIR`:

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

`better_human.io.models_dir()` is the one reader of the variable; it raises `AssetNotFoundError` naming the variable
when it is unset or names no folder, and a loader raises naming the missing file.

```python
from better_human import io

print(io.models_dir().is_dir())
```
```output
True
```

## Load a family with its defaults

`bh.load(family)` runs the family's loader with its defaults and builds the `HumanModel`. Every family gives a
BetterRobot model with a free-flyer root; the joint kinds under it are the family's own (ball joints for the SMPL
family and SOMA, chains of revolute joints for MHR).

```python
import better_human as bh

for family in ("smpl", "mhr", "soma"):
    human = bh.load(family)
    joints, vertices = len(human.spec.joint_names), human.spec.num_vertices
    print(f"{family} nq={human.model.nq} nv={human.model.nv} joints={joints} vertices={vertices}")
```
```output
smpl nq=99 nv=75 joints=24 vertices=6890
mhr nq=132 nv=131 joints=127 vertices=18439
soma nq=311 nv=234 joints=110 vertices=18056
```

A file outside the layout is passed as the path: `bh.load("smpl", "/data/SMPL_MALE.npz")`. An explicit path always
wins over the folder.

## Choose dtype and device at load

Dtype and device are chosen once, by the loader; no BetterHuman object has `to`, and a tensor in
another dtype than the model's raises instead of being cast.

```python
import torch

human = bh.load("smpl", dtype=torch.float64, device="cpu")
print(human.model.q_neutral.dtype, human.model.q_neutral.device)
```
```output
torch.float64 cpu
```

## Pass family options

Gender, the number of shape coefficients, the level of detail and the partial-model switches are the loader's
options; `HumanModel.from_spec` takes the loader's `FamilySpec` and the options of the body. `hands=False` and `face=False` load SMPL-X without its hand
and face joints and without its expression group:

```python
from better_human import HumanModel, families

spec = families.smplx(gender="female", num_betas=10, hands=False, face=False)
human = HumanModel.from_spec(spec, density=None)
print(len(spec.joint_names), human.model.nq, dict(spec.identity_bases)["betas"].shape[-1], sorted(spec.stream_bases))
```
```output
22 91 10 []
```

`density=None` builds the body without masses; the default is the `"Dempster"` density table, and a float is a uniform density in kg/m³. MHR takes
`families.mhr(lod=0)` to `lod=6`; MANO takes `side="left"`; `num_pca_comps=` fixes the hand pose width of SMPL-X and
MANO at load. `families.drop_joints(spec, names)` removes any other terminal joint subtree of an SMPL-family spec.

## Attach contact frames

A contact frame is a named point on the mesh, rigid with a joint; `contact_frames` maps a name to `(vertex_id,
joint_name)`, and `None` as the joint means the vertex's strongest skinning joint. `io.read_frame_set(path)` reads
the mapping from a frame-set file (the studio writes them), and `io.shipped_frame_set("smplx_contacts")` returns the
35 SMPL-X contact points BetterHuman ships:

```python
frames = io.shipped_frame_set("smplx_contacts")
human = bh.load("smplx", contact_frames=frames)
print(len(frames), human.model.nframes, list(frames)[:3])
```
```output
35 91 ['heel_left', 'heel_right', 'toe_above_left']
```
