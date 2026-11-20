# Families and files

A family is data. Its loader reads the weight file and returns a `FamilySpec`: the template mesh, the skinning
weights, the joint tree, the blend shapes, the pose corrective and the body parts. `HumanModel.from_spec` builds the
body from the spec, with the options of the body (density, contact frames). `bh.load(family)` does both with the
family's defaults. Every family gives a BetterRobot model with a free-flyer root; the joints under it are the
family's own (ball joints for the SMPL family and SOMA, chains of revolute joints for MHR).

The names you will use:

| Name | What it does |
|---|---|
| `bh.load(family, path=None, *, dtype=, device=, density=, contact_frames=)` | the body with the family's defaults |
| `families.smpl`, `smplh`, `smplx`, `mano`, `flame`, `mhr`, `soma` | each family's loader and its options (gender, `num_betas`, `lod`, `hands=`, `face=`) |
| `HumanModel.from_spec(spec, *, density=, contact_frames=)` | the body from a loader's spec |
| `families.drop_joints(spec, names)` | the spec without some end joints, their skin moved to the parent |
| `io.read_frame_set`, `io.write_frame_set`, `io.shipped_frame_set` | contact frames as files |
| `bh.from_classic`, `bh.to_classic` | the family's own parameters (`betas`, `body_pose`, ...) to and from `q` |
| `io.mhr_to_smplx_mapping()` | the MHR triangle and barycentric weights of each SMPL-X vertex, from MHR's authors |

Dtype and device are chosen once, at load: `bh.load("smplx", dtype=torch.float64, device="cuda")`. No BetterHuman
object has a `to` method, and a tensor in another dtype raises instead of being cast.

## Family options

The loader takes the options of the file: gender, the number of shape coefficients, the level of detail (MHR), and
switches for partial bodies. Here SMPL-X loads without its hands and face, so the expression group goes too:

```python
import torch
import better_human as bh
from better_human import HumanModel, families

SMPLX = "tests/support/random_weights/smplx_random.npz"  # random weights in SMPL-X's layout
spec = families.smplx(SMPLX, hands=False, face=False)
human = HumanModel.from_spec(spec, density=None)  # no masses
print(len(spec.joint_names), human.model.nq)
print(spec.identity_groups, spec.stream_groups)
```
```output
22 91
{'betas': 16} {}
```

## Contact frames

A contact frame is a named point on the mesh, rigid with a joint: `name -> (vertex_id, joint_name)`, where `None`
as the joint means the vertex's strongest skinning joint. Frames become frames of the BetterRobot model, so
Jacobians and dynamics see them. BetterHuman ships the 35 SMPL-X contact points the studio placed:

```python
frames = bh.io.shipped_frame_set("smplx_contacts")
human = bh.load("smplx", SMPLX, contact_frames=frames)
state = human.bake().forward(human.model.q_neutral.expand(8, -1))  # 8 poses
print(list(frames)[:3])
print(state.contact_frame_poses(frames).shape)  # (8, 35, 7): [tx, ty, tz, qx, qy, qz, qw]
```
```output
['heel_left', 'heel_right', 'toe_above_left']
torch.Size([8, 35, 7])
```

`io.write_frame_set(path, frames)` and `io.read_frame_set(path)` keep a frame set in a versioned JSON file. Every
file BetterHuman reads or writes names its format and version; [the frame-set
format](reference/formats/frame_set.md) is one of seven.

## The family's own parameters

Inside BetterHuman there is one pose convention, BetterRobot's `q`. A family's own parameters (`betas`,
`global_orient`, `body_pose`, MHR's `model_parameters`, ...) enter and leave in one place, a `<Family>Classic`
record, and the round trip is exact to float precision:

```python
human = bh.load("smplx", SMPLX)
body_pose = torch.full((63,), 0.1)  # 21 body joints, axis-angle
identity, q, streams = bh.from_classic(human, families.SMPLXClassic(betas=torch.zeros(16), body_pose=body_pose))
back = bh.to_classic(identity, q, streams=streams)
print(type(back).__name__, q.shape, torch.allclose(back.body_pose, body_pose))
```
```output
SMPLXClassic torch.Size([223]) True
```

See also: [Bodies](bodies.md), which poses what you loaded.
