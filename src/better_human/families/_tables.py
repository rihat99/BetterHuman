"""Static per-family tables: joint names, density categories, landmark vertex ids, subtree index groups, file names."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True)
class FamilyTable:
    """Authored metadata of one SMPL-family model (parents come from the weight file)."""

    name: str
    joint_names: tuple[str, ...]
    file_pattern: str  #: file under ``$BETTERHUMAN_MODELS_DIR/<folder>/``; ``{gender}`` / ``{side}`` placeholders.
    folder: str
    #: Vertices of the authored mesh: the loader reads a file of exactly this many, so the shipped landmarks, contact
    #: frames and segmentation index the right vertices.
    num_vertices: int
    #: The density category of each joint's part: one of the shipped tables' categories, which
    #: are SMPL's 24 part names; ``{side}`` is MANO's side. Fingers take the hand, face joints and eyes the head.
    density_categories: tuple[str, ...]
    keypoint_names: tuple[str, ...] = ()
    keypoint_vertex_ids: tuple[int, ...] = ()
    #: Terminal subtrees that can be dropped at load, by native joint index.
    face_joints: tuple[int, ...] = ()
    left_hand_joints: tuple[int, ...] = ()
    right_hand_joints: tuple[int, ...] = ()
    #: Shipped closed segmentation (``io.mass_data.shipped_segmentation``) whose parts are this family's joints.
    closed_segmentation: str | None = None


KEYPOINT_NAMES = (
    "nose",
    "reye",
    "leye",
    "rear",
    "lear",
    "rthumb",
    "rindex",
    "rmiddle",
    "rring",
    "rpinky",
    "lthumb",
    "lindex",
    "lmiddle",
    "lring",
    "lpinky",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
)

SMPL_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

SMPL = FamilyTable(
    name="smpl",
    joint_names=SMPL_JOINT_NAMES,
    file_pattern="SMPL_{gender}.npz",
    folder="smpl",
    num_vertices=6890,
    density_categories=SMPL_JOINT_NAMES,
    keypoint_names=KEYPOINT_NAMES,
    keypoint_vertex_ids=(
        332,
        6260,
        2800,
        4071,
        583,
        6191,
        5782,
        5905,
        6016,
        6133,
        2746,
        2319,
        2445,
        2556,
        2673,
        3216,
        3226,
        3387,
        6617,
        6624,
        6787,
    ),
    closed_segmentation="smpl",
)

SMPLH = FamilyTable(
    name="smplh",
    joint_names=SMPL.joint_names[:22]
    + (
        "left_index1",
        "left_index2",
        "left_index3",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",
    )
    + (
        "right_index1",
        "right_index2",
        "right_index3",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3",
    ),
    file_pattern="smplh/{gender_lower}/model.npz",
    folder="smplh",
    num_vertices=6890,
    density_categories=SMPL_JOINT_NAMES[:22] + ("left_hand",) * 15 + ("right_hand",) * 15,
    keypoint_names=KEYPOINT_NAMES,
    keypoint_vertex_ids=SMPL.keypoint_vertex_ids,
    left_hand_joints=tuple(range(22, 37)),
    right_hand_joints=tuple(range(37, 52)),
)

SMPLX = FamilyTable(
    name="smplx",
    joint_names=(
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "jaw",
        "left_eye_smplhf",
        "right_eye_smplhf",
        "left_index1",
        "left_index2",
        "left_index3",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",
        "right_index1",
        "right_index2",
        "right_index3",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3",
    ),
    file_pattern="SMPLX_{gender}.npz",
    folder="smplx",
    num_vertices=10475,
    density_categories=SMPL_JOINT_NAMES[:22] + ("head",) * 3 + ("left_hand",) * 15 + ("right_hand",) * 15,
    keypoint_names=KEYPOINT_NAMES,
    keypoint_vertex_ids=(
        9120,
        9929,
        9448,
        616,
        6,
        8079,
        7669,
        7794,
        7905,
        8022,
        5361,
        4933,
        5058,
        5169,
        5286,
        5770,
        5780,
        8846,
        8463,
        8474,
        8635,
    ),
    face_joints=(22, 23, 24),
    left_hand_joints=tuple(range(25, 40)),
    right_hand_joints=tuple(range(40, 55)),
)

MANO = FamilyTable(
    name="mano",
    joint_names=(
        "wrist",
        "index1",
        "index2",
        "index3",
        "middle1",
        "middle2",
        "middle3",
        "pinky1",
        "pinky2",
        "pinky3",
        "ring1",
        "ring2",
        "ring3",
        "thumb1",
        "thumb2",
        "thumb3",
    ),
    file_pattern="MANO_{side}.npz",
    folder="mano",
    num_vertices=778,
    density_categories=("{side}_wrist",) + ("{side}_hand",) * 15,
)

FLAME = FamilyTable(
    name="flame",
    joint_names=("global", "neck", "jaw", "left_eye", "right_eye"),
    file_pattern="{gender}_model.npz",
    folder="flame",
    num_vertices=5023,
    density_categories=("head", "neck", "head", "head", "head"),  # FLAME's root part "global" is the head
)

__all__ = ["FLAME", "KEYPOINT_NAMES", "MANO", "SMPL", "SMPLH", "SMPLX", "SMPL_JOINT_NAMES", "FamilyTable"]
