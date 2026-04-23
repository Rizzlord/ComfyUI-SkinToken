from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from omegaconf import OmegaConf

SKELETON_TEMPLATE_KEEP = "original"
SKELETON_TEMPLATE_MIXAMO = "mixamo"
SKELETON_TEMPLATE_UE5 = "ue5"

SKELETON_TEMPLATE_KEYS = [
    SKELETON_TEMPLATE_KEEP,
    SKELETON_TEMPLATE_MIXAMO,
    SKELETON_TEMPLATE_UE5,
]

SKELETON_TEMPLATE_LABELS = {
    SKELETON_TEMPLATE_KEEP: "Keep model names",
    SKELETON_TEMPLATE_MIXAMO: "Mixamo",
    SKELETON_TEMPLATE_UE5: "Unreal Engine 5",
}

SKELETON_TEMPLATE_LABEL_CHOICES = [
    SKELETON_TEMPLATE_LABELS[key] for key in SKELETON_TEMPLATE_KEYS
]

_LABEL_TO_TEMPLATE = {
    label: key for (key, label) in SKELETON_TEMPLATE_LABELS.items()
}

_TEMPLATE_CONFIGS = {
    SKELETON_TEMPLATE_MIXAMO: (
        Path(__file__).resolve().parents[2] / "configs" / "skeleton" / "mixamo.yaml"
    ),
    SKELETON_TEMPLATE_UE5: (
        Path(__file__).resolve().parents[2] / "configs" / "skeleton" / "ue5.yaml"
    ),
}

_GENERIC_BONE_NAME = re.compile(r"^(bone|joint)[_ .:-]*\d+$", re.IGNORECASE)
_EXTRA_BONE_PREFIX = {
    SKELETON_TEMPLATE_MIXAMO: "mixamorig:Extra",
    SKELETON_TEMPLATE_UE5: "extra",
}

_SEMANTIC_TEMPLATE_NAMES = {
    SKELETON_TEMPLATE_MIXAMO: {
        "hips": "mixamorig:Hips",
        "spines": [
            "mixamorig:Spine",
            "mixamorig:Spine1",
            "mixamorig:Spine2",
        ],
        "neck": "mixamorig:Neck",
        "head": "mixamorig:Head",
        "left_arm": [
            "mixamorig:LeftShoulder",
            "mixamorig:LeftArm",
            "mixamorig:LeftForeArm",
            "mixamorig:LeftHand",
        ],
        "right_arm": [
            "mixamorig:RightShoulder",
            "mixamorig:RightArm",
            "mixamorig:RightForeArm",
            "mixamorig:RightHand",
        ],
        "left_leg": [
            "mixamorig:LeftUpLeg",
            "mixamorig:LeftLeg",
            "mixamorig:LeftFoot",
            "mixamorig:LeftToeBase",
        ],
        "right_leg": [
            "mixamorig:RightUpLeg",
            "mixamorig:RightLeg",
            "mixamorig:RightFoot",
            "mixamorig:RightToeBase",
        ],
        "left_fingers": {
            "thumb": [
                "mixamorig:LeftHandThumb1",
                "mixamorig:LeftHandThumb2",
                "mixamorig:LeftHandThumb3",
            ],
            "index": [
                "mixamorig:LeftHandIndex1",
                "mixamorig:LeftHandIndex2",
                "mixamorig:LeftHandIndex3",
            ],
            "middle": [
                "mixamorig:LeftHandMiddle1",
                "mixamorig:LeftHandMiddle2",
                "mixamorig:LeftHandMiddle3",
            ],
            "ring": [
                "mixamorig:LeftHandRing1",
                "mixamorig:LeftHandRing2",
                "mixamorig:LeftHandRing3",
            ],
            "pinky": [
                "mixamorig:LeftHandPinky1",
                "mixamorig:LeftHandPinky2",
                "mixamorig:LeftHandPinky3",
            ],
        },
        "right_fingers": {
            "thumb": [
                "mixamorig:RightHandThumb1",
                "mixamorig:RightHandThumb2",
                "mixamorig:RightHandThumb3",
            ],
            "index": [
                "mixamorig:RightHandIndex1",
                "mixamorig:RightHandIndex2",
                "mixamorig:RightHandIndex3",
            ],
            "middle": [
                "mixamorig:RightHandMiddle1",
                "mixamorig:RightHandMiddle2",
                "mixamorig:RightHandMiddle3",
            ],
            "ring": [
                "mixamorig:RightHandRing1",
                "mixamorig:RightHandRing2",
                "mixamorig:RightHandRing3",
            ],
            "pinky": [
                "mixamorig:RightHandPinky1",
                "mixamorig:RightHandPinky2",
                "mixamorig:RightHandPinky3",
            ],
        },
    },
    SKELETON_TEMPLATE_UE5: {
        "hips": "pelvis",
        "spines": [
            "spine_01",
            "spine_02",
            "spine_03",
        ],
        "neck": "neck_01",
        "head": "head",
        "left_arm": [
            "clavicle_l",
            "upperarm_l",
            "lowerarm_l",
            "hand_l",
        ],
        "right_arm": [
            "clavicle_r",
            "upperarm_r",
            "lowerarm_r",
            "hand_r",
        ],
        "left_leg": [
            "thigh_l",
            "calf_l",
            "foot_l",
            "ball_l",
        ],
        "right_leg": [
            "thigh_r",
            "calf_r",
            "foot_r",
            "ball_r",
        ],
        "left_fingers": {
            "thumb": ["thumb_01_l", "thumb_02_l", "thumb_03_l"],
            "index": ["index_01_l", "index_02_l", "index_03_l"],
            "middle": ["middle_01_l", "middle_02_l", "middle_03_l"],
            "ring": ["ring_01_l", "ring_02_l", "ring_03_l"],
            "pinky": ["pinky_01_l", "pinky_02_l", "pinky_03_l"],
        },
        "right_fingers": {
            "thumb": ["thumb_01_r", "thumb_02_r", "thumb_03_r"],
            "index": ["index_01_r", "index_02_r", "index_03_r"],
            "middle": ["middle_01_r", "middle_02_r", "middle_03_r"],
            "ring": ["ring_01_r", "ring_02_r", "ring_03_r"],
            "pinky": ["pinky_01_r", "pinky_02_r", "pinky_03_r"],
        },
    },
}


def normalize_skeleton_template(template: Optional[str]) -> str:
    if template is None:
        return SKELETON_TEMPLATE_KEEP
    template = _LABEL_TO_TEMPLATE.get(template, template)
    if template not in SKELETON_TEMPLATE_KEYS:
        raise ValueError(f"Unknown skeleton template: {template}")
    return template


@lru_cache(maxsize=None)
def _load_template_names(template: str) -> List[str]:
    template = normalize_skeleton_template(template)
    if template == SKELETON_TEMPLATE_KEEP:
        return []
    config = OmegaConf.load(_TEMPLATE_CONFIGS[template])
    names: List[str] = []
    for part in config.parts_order:
        names.extend(str(name) for name in config.parts[part])
    return names


def _is_generic_bone_name(name: str) -> bool:
    return bool(_GENERIC_BONE_NAME.fullmatch(name))


def _resolve_source_names(
    joint_names: Optional[Iterable[str]],
    joint_count: int,
) -> List[str]:
    names = list(joint_names or [])
    if len(names) < joint_count:
        names.extend(f"bone_{i}" for i in range(len(names), joint_count))
    return names[:joint_count]


def _make_unique_name(name: str, seen: set[str]) -> str:
    if name not in seen:
        seen.add(name)
        return name
    suffix = 1
    candidate = f"{name}_{suffix}"
    while candidate in seen:
        suffix += 1
        candidate = f"{name}_{suffix}"
    seen.add(candidate)
    return candidate


def _make_unique_names(names: List[str]) -> List[str]:
    seen: set[str] = set()
    return [_make_unique_name(name, seen) for name in names]


def _build_children(parents: np.ndarray) -> tuple[int, List[List[int]]]:
    root = -1
    children = [[] for _ in range(len(parents))]
    for idx, parent in enumerate(parents.tolist()):
        if parent == -1:
            if root != -1:
                raise ValueError("Multiple roots found in skeleton.")
            root = idx
            continue
        children[parent].append(idx)
    if root == -1:
        raise ValueError("Skeleton has no root.")
    return root, children


def _post_order(root: int, children: List[List[int]]) -> List[int]:
    order: List[int] = []
    stack = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            order.append(node)
            continue
        stack.append((node, True))
        for child in children[node]:
            stack.append((child, False))
    return order


def _compute_subtree_stats(
    joints: np.ndarray,
    children: List[List[int]],
    root: int,
    x_axis: int,
    z_axis: int,
) -> dict[str, np.ndarray]:
    joint_count = joints.shape[0]
    subtree_size = np.ones(joint_count, dtype=np.int32)
    subtree_min_z = joints[:, z_axis].copy()
    subtree_max_z = joints[:, z_axis].copy()
    subtree_min_x = joints[:, x_axis].copy()
    subtree_max_x = joints[:, x_axis].copy()

    for node in _post_order(root=root, children=children):
        for child in children[node]:
            subtree_size[node] += subtree_size[child]
            subtree_min_z[node] = min(subtree_min_z[node], subtree_min_z[child])
            subtree_max_z[node] = max(subtree_max_z[node], subtree_max_z[child])
            subtree_min_x[node] = min(subtree_min_x[node], subtree_min_x[child])
            subtree_max_x[node] = max(subtree_max_x[node], subtree_max_x[child])

    return {
        "size": subtree_size,
        "min_z": subtree_min_z,
        "max_z": subtree_max_z,
        "min_x": subtree_min_x,
        "max_x": subtree_max_x,
    }


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v)
    return v / norm


def _principal_axis(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((points.shape[1],), dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    if np.linalg.norm(centered) < 1e-8:
        return np.zeros((points.shape[1],), dtype=np.float32)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return vh[0]


def _side_sign(side: str, left_is_positive_x: bool) -> float:
    if left_is_positive_x:
        return 1.0 if side == "left" else -1.0
    return -1.0 if side == "left" else 1.0


def _build_extra_names(
    source_names: List[str],
    template: str,
) -> List[str]:
    extra_prefix = _EXTRA_BONE_PREFIX[template]
    fallback = []
    for idx, name in enumerate(source_names):
        if name and not _is_generic_bone_name(name):
            fallback.append(name)
        else:
            fallback.append(f"{extra_prefix}_{idx:02d}")
    return fallback


def _follow_best_chain(
    start: int,
    children: List[List[int]],
    joints: np.ndarray,
    root_x: float,
    x_axis: int,
    z_axis: int,
    subtree_stats: dict[str, np.ndarray],
    mode: str,
    side: Optional[str] = None,
    left_is_positive_x: bool = True,
) -> List[int]:
    chain = [start]
    current = start
    seen = {start}
    while True:
        child_nodes = [child for child in children[current] if child not in seen]
        if not child_nodes:
            break
        if mode == "arm" and len(child_nodes) >= 3 and len(chain) >= 3:
            break

        best_child = None
        best_score = float("-inf")
        for child in child_nodes:
            x_offset = joints[child, x_axis] - root_x
            z_offset = joints[child, z_axis] - joints[current, z_axis]
            side_bonus = 0.0
            if side is not None:
                side_dir = _side_sign(side, left_is_positive_x=left_is_positive_x)
                side_bonus = side_dir * x_offset
            if mode == "spine":
                score = (
                    5.0 * (subtree_stats["max_z"][child] - joints[current, z_axis])
                    + 2.0 * z_offset
                    - 3.0 * abs(x_offset)
                    + 0.02 * float(subtree_stats["size"][child])
                )
            elif mode == "arm":
                score = (
                    5.0 * abs(x_offset)
                    + 1.5 * side_bonus
                    + 1.0 * (subtree_stats["max_z"][child] - joints[current, z_axis])
                    + 0.02 * float(subtree_stats["size"][child])
                )
            else:
                score = (
                    5.0 * (joints[current, z_axis] - subtree_stats["min_z"][child])
                    + 1.5 * abs(x_offset)
                    + 1.0 * side_bonus
                    + 0.02 * float(subtree_stats["size"][child])
                )
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            break
        if mode == "spine" and best_score < 0.05:
            break
        chain.append(best_child)
        seen.add(best_child)
        current = best_child
    return chain


def _select_side_branch(
    candidates: List[int],
    joints: np.ndarray,
    root_x: float,
    x_axis: int,
    z_axis: int,
    subtree_stats: dict[str, np.ndarray],
    side: str,
    kind: str,
    left_is_positive_x: bool,
) -> Optional[int]:
    best_node = None
    best_score = float("-inf")
    side_dir = _side_sign(side, left_is_positive_x=left_is_positive_x)
    for node in candidates:
        x_offset = joints[node, x_axis] - root_x
        if side_dir * x_offset <= 0:
            continue
        lateral = max(
            abs(subtree_stats["max_x"][node] - root_x),
            abs(subtree_stats["min_x"][node] - root_x),
        )
        if kind == "arm":
            score = (
                6.0 * lateral
                + 1.5 * (subtree_stats["max_z"][node] - joints[:, z_axis].min())
                + 0.05 * float(subtree_stats["size"][node])
            )
        else:
            score = (
                5.0 * (joints[:, z_axis].max() - subtree_stats["min_z"][node])
                + 2.0 * lateral
                + 0.05 * float(subtree_stats["size"][node])
            )
        if score > best_score:
            best_score = score
            best_node = node
    return best_node


def _select_top_branch_candidates(
    candidates: List[int],
    joints: np.ndarray,
    root_x: float,
    x_axis: int,
    z_axis: int,
    subtree_stats: dict[str, np.ndarray],
    kind: str,
    limit: int,
) -> List[int]:
    scored: List[tuple[float, int]] = []
    for node in candidates:
        x_offset = joints[node, x_axis] - root_x
        lateral = max(
            abs(subtree_stats["max_x"][node] - root_x),
            abs(subtree_stats["min_x"][node] - root_x),
        )
        if kind == "arm":
            score = (
                6.0 * lateral
                + 1.5 * (subtree_stats["max_z"][node] - joints[:, z_axis].min())
                + 0.05 * float(subtree_stats["size"][node])
            )
        else:
            score = (
                5.0 * (joints[:, z_axis].max() - subtree_stats["min_z"][node])
                + 2.0 * abs(x_offset)
                + 0.05 * float(subtree_stats["size"][node])
            )
        scored.append((score, node))
    scored.sort(reverse=True)
    return [node for _, node in scored[:limit]]


def _infer_left_positive_x(
    joints: np.ndarray,
    root_idx: int,
    central_chain: List[int],
    leg_chains: List[List[int]],
    z_axis: int,
) -> Optional[bool]:
    if len(central_chain) < 2:
        return None
    up_vector = _normalize(joints[central_chain[-1]] - joints[root_idx])
    if np.linalg.norm(up_vector) < 1e-6:
        up_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    forward_vectors = []
    for chain in leg_chains:
        if len(chain) >= 2:
            direction = joints[chain[-1]] - joints[chain[-2]]
            direction = direction - np.dot(direction, up_vector) * up_vector
            if np.linalg.norm(direction) > 1e-6:
                forward_vectors.append(_normalize(direction))
    if not forward_vectors:
        return None

    forward_vector = _normalize(np.mean(forward_vectors, axis=0))
    if np.linalg.norm(forward_vector) < 1e-6:
        return None

    left_axis = _normalize(np.cross(up_vector, forward_vector))
    if np.linalg.norm(left_axis) < 1e-6:
        return None
    return bool(left_axis[0] >= 0.0)


def _assign_chain_names(
    target_names: List[str],
    chain: List[int],
    semantic_names: List[str],
) -> None:
    for node, name in zip(chain, semantic_names):
        target_names[node] = name


def _extend_linear_branch(
    start: int,
    children: List[List[int]],
    joints: np.ndarray,
    hand_idx: int,
    max_length: int = 3,
) -> List[int]:
    chain = [start]
    current = start
    seen = {start}
    while len(chain) < max_length:
        child_nodes = [child for child in children[current] if child not in seen]
        if not child_nodes:
            break
        if len(child_nodes) == 1:
            best_child = child_nodes[0]
        else:
            best_child = max(
                child_nodes,
                key=lambda child: float(np.linalg.norm(joints[child] - joints[hand_idx])),
            )
        chain.append(best_child)
        seen.add(best_child)
        current = best_child
    return chain


def _classify_finger_branches(
    hand_idx: int,
    children: List[List[int]],
    joints: np.ndarray,
) -> List[List[int]]:
    branches = [
        _extend_linear_branch(start=child, children=children, joints=joints, hand_idx=hand_idx)
        for child in children[hand_idx]
    ]
    if len(branches) <= 5:
        return branches
    branches.sort(
        key=lambda branch: (
            len(branch),
            float(np.linalg.norm(joints[branch[-1]] - joints[hand_idx])),
        ),
        reverse=True,
    )
    return branches[:5]


def _assign_finger_names(
    target_names: List[str],
    hand_idx: Optional[int],
    side: str,
    template: str,
    children: List[List[int]],
    joints: np.ndarray,
) -> None:
    if hand_idx is None:
        return
    finger_branches = _classify_finger_branches(hand_idx=hand_idx, children=children, joints=joints)
    if len(finger_branches) < 2:
        return

    tip_dirs = np.stack(
        [
            _normalize(joints[branch[-1]] - joints[hand_idx])
            for branch in finger_branches
        ],
        axis=0,
    )

    thumb_idx = 0
    if len(finger_branches) >= 3:
        cosine = tip_dirs @ tip_dirs.T
        denom = max(len(finger_branches) - 1, 1)
        avg_similarity = (cosine.sum(axis=1) - 1.0) / denom
        thumb_idx = int(np.argmin(avg_similarity))

    thumb_branch = finger_branches[thumb_idx]
    other_items = [
        (idx, branch)
        for idx, branch in enumerate(finger_branches)
        if idx != thumb_idx
    ]
    if not other_items:
        return

    other_branches = [branch for _, branch in other_items]
    forward = _normalize(np.mean(
        [_normalize(joints[branch[-1]] - joints[hand_idx]) for branch in other_branches],
        axis=0,
    ))
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    base_positions = np.stack(
        [joints[branch[0]] - joints[hand_idx] for branch in other_branches],
        axis=0,
    )
    projected = base_positions - np.outer(base_positions @ forward, forward)
    lateral_axis = _principal_axis(projected)
    if np.linalg.norm(lateral_axis) < 1e-6:
        lateral_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    other_scalars = projected @ lateral_axis
    thumb_base = joints[thumb_branch[0]] - joints[hand_idx]
    thumb_projected = thumb_base - np.dot(thumb_base, forward) * forward
    thumb_scalar = float(np.dot(thumb_projected, lateral_axis))

    ordered = list(zip(other_scalars.tolist(), other_branches))
    ordered.sort(key=lambda item: item[0])
    if thumb_scalar >= float(np.mean(other_scalars)):
        ordered.reverse()

    semantic = _SEMANTIC_TEMPLATE_NAMES[template][f"{side}_fingers"]
    _assign_chain_names(target_names, thumb_branch, semantic["thumb"])
    semantic_order = ["index", "middle", "ring", "pinky"]
    for key, (_, branch) in zip(semantic_order, ordered):
        _assign_chain_names(target_names, branch, semantic[key])


def _build_humanoid_template_names(
    source_names: List[str],
    joints: np.ndarray,
    parents: np.ndarray,
    template: str,
    left_is_positive_x: bool,
) -> tuple[Optional[List[str]], int]:
    if template not in _SEMANTIC_TEMPLATE_NAMES:
        return None, -1

    x_axis = 0
    z_axis = 2

    try:
        root, children = _build_children(parents=parents)
    except ValueError:
        return None, -1

    if joints.shape[0] < 10:
        return None, -1

    root_x = float(joints[root, x_axis])
    subtree_stats = _compute_subtree_stats(
        joints=joints,
        children=children,
        root=root,
        x_axis=x_axis,
        z_axis=z_axis,
    )

    root_children = children[root]
    if not root_children:
        return None, -1

    spine_child = None
    spine_score = float("-inf")
    for child in root_children:
        score = (
            6.0 * (subtree_stats["max_z"][child] - joints[root, z_axis])
            - 4.0 * abs(joints[child, x_axis] - root_x)
            + 0.05 * float(subtree_stats["size"][child])
        )
        if score > spine_score:
            spine_score = score
            spine_child = child
    if spine_child is None:
        return None, -1

    central_chain = [root]
    central_chain.extend(
        _follow_best_chain(
            start=spine_child,
            children=children,
            joints=joints,
            root_x=root_x,
            x_axis=x_axis,
            z_axis=z_axis,
            subtree_stats=subtree_stats,
            mode="spine",
            left_is_positive_x=left_is_positive_x,
        )
    )
    central_set = set(central_chain)
    if len(central_chain) < 4:
        return None, -1

    central_next = {central_chain[idx]: central_chain[idx + 1] for idx in range(len(central_chain) - 1)}

    arm_candidates: List[int] = []
    leg_candidates: List[int] = []
    for center in central_chain:
        for child in children[center]:
            if central_next.get(center) == child:
                continue
            if center == root:
                leg_candidates.append(child)
            else:
                arm_candidates.append(child)

    semantic_names = _build_extra_names(source_names=source_names, template=template)
    slots = _SEMANTIC_TEMPLATE_NAMES[template]

    semantic_names[root] = slots["hips"]

    torso_nodes = central_chain[1:]
    if len(torso_nodes) >= 2:
        semantic_names[torso_nodes[-1]] = slots["head"]
        semantic_names[torso_nodes[-2]] = slots["neck"]
    spine_nodes = torso_nodes[:-2] if len(torso_nodes) > 2 else torso_nodes[:-1]
    _assign_chain_names(semantic_names, spine_nodes, slots["spines"])

    preview_leg_roots = _select_top_branch_candidates(
        candidates=leg_candidates,
        joints=joints,
        root_x=root_x,
        x_axis=x_axis,
        z_axis=z_axis,
        subtree_stats=subtree_stats,
        kind="leg",
        limit=2,
    )
    preview_leg_chains = [
        _follow_best_chain(
            start=leg_root,
            children=children,
            joints=joints,
            root_x=root_x,
            x_axis=x_axis,
            z_axis=z_axis,
            subtree_stats=subtree_stats,
            mode="leg",
            left_is_positive_x=left_is_positive_x,
        )
        for leg_root in preview_leg_roots
    ]
    inferred_left_positive_x = _infer_left_positive_x(
        joints=joints,
        root_idx=root,
        central_chain=central_chain,
        leg_chains=preview_leg_chains,
        z_axis=z_axis,
    )
    resolved_left_positive_x = (
        left_is_positive_x if inferred_left_positive_x is None else inferred_left_positive_x
    )

    left_arm_root = _select_side_branch(
        candidates=arm_candidates,
        joints=joints,
        root_x=root_x,
        x_axis=x_axis,
        z_axis=z_axis,
        subtree_stats=subtree_stats,
        side="left",
        kind="arm",
        left_is_positive_x=resolved_left_positive_x,
    )
    right_arm_root = _select_side_branch(
        candidates=arm_candidates,
        joints=joints,
        root_x=root_x,
        x_axis=x_axis,
        z_axis=z_axis,
        subtree_stats=subtree_stats,
        side="right",
        kind="arm",
        left_is_positive_x=resolved_left_positive_x,
    )
    left_leg_root = _select_side_branch(
        candidates=leg_candidates,
        joints=joints,
        root_x=root_x,
        x_axis=x_axis,
        z_axis=z_axis,
        subtree_stats=subtree_stats,
        side="left",
        kind="leg",
        left_is_positive_x=resolved_left_positive_x,
    )
    right_leg_root = _select_side_branch(
        candidates=leg_candidates,
        joints=joints,
        root_x=root_x,
        x_axis=x_axis,
        z_axis=z_axis,
        subtree_stats=subtree_stats,
        side="right",
        kind="leg",
        left_is_positive_x=resolved_left_positive_x,
    )

    left_arm_chain = (
        _follow_best_chain(
            start=left_arm_root,
            children=children,
            joints=joints,
            root_x=root_x,
            x_axis=x_axis,
            z_axis=z_axis,
            subtree_stats=subtree_stats,
            mode="arm",
            side="left",
            left_is_positive_x=resolved_left_positive_x,
        )
        if left_arm_root is not None
        else []
    )
    right_arm_chain = (
        _follow_best_chain(
            start=right_arm_root,
            children=children,
            joints=joints,
            root_x=root_x,
            x_axis=x_axis,
            z_axis=z_axis,
            subtree_stats=subtree_stats,
            mode="arm",
            side="right",
            left_is_positive_x=resolved_left_positive_x,
        )
        if right_arm_root is not None
        else []
    )
    left_leg_chain = (
        _follow_best_chain(
            start=left_leg_root,
            children=children,
            joints=joints,
            root_x=root_x,
            x_axis=x_axis,
            z_axis=z_axis,
            subtree_stats=subtree_stats,
            mode="leg",
            side="left",
            left_is_positive_x=resolved_left_positive_x,
        )
        if left_leg_root is not None
        else []
    )
    right_leg_chain = (
        _follow_best_chain(
            start=right_leg_root,
            children=children,
            joints=joints,
            root_x=root_x,
            x_axis=x_axis,
            z_axis=z_axis,
            subtree_stats=subtree_stats,
            mode="leg",
            side="right",
            left_is_positive_x=resolved_left_positive_x,
        )
        if right_leg_root is not None
        else []
    )

    _assign_chain_names(semantic_names, left_arm_chain, slots["left_arm"])
    _assign_chain_names(semantic_names, right_arm_chain, slots["right_arm"])
    _assign_chain_names(semantic_names, left_leg_chain, slots["left_leg"])
    _assign_chain_names(semantic_names, right_leg_chain, slots["right_leg"])

    left_hand_idx = left_arm_chain[min(3, len(left_arm_chain) - 1)] if left_arm_chain else None
    right_hand_idx = right_arm_chain[min(3, len(right_arm_chain) - 1)] if right_arm_chain else None
    _assign_finger_names(
        target_names=semantic_names,
        hand_idx=left_hand_idx,
        side="left",
        template=template,
        children=children,
        joints=joints,
    )
    _assign_finger_names(
        target_names=semantic_names,
        hand_idx=right_hand_idx,
        side="right",
        template=template,
        children=children,
        joints=joints,
    )

    found_limbs = sum(bool(chain) for chain in [left_arm_chain, right_arm_chain, left_leg_chain, right_leg_chain])
    if found_limbs < 3:
        return None, found_limbs
    score = (
        found_limbs * 10
        + min(len(left_arm_chain), 4)
        + min(len(right_arm_chain), 4)
        + min(len(left_leg_chain), 4)
        + min(len(right_leg_chain), 4)
        + min(len(central_chain), 6)
    )
    return _make_unique_names(semantic_names), score


def _apply_humanoid_template(
    source_names: List[str],
    joints: np.ndarray,
    parents: np.ndarray,
    template: str,
) -> Optional[List[str]]:
    best_names = None
    best_score = -1
    for left_is_positive_x in (True, False):
        candidate_names, candidate_score = _build_humanoid_template_names(
            source_names=source_names,
            joints=joints,
            parents=parents,
            template=template,
            left_is_positive_x=left_is_positive_x,
        )
        if candidate_names is not None and candidate_score > best_score:
            best_names = candidate_names
            best_score = candidate_score
    return best_names


def apply_joint_name_template(
    joint_names: Optional[Iterable[str]],
    joint_count: int,
    template: Optional[str],
) -> List[str]:
    template = normalize_skeleton_template(template)
    source_names = _resolve_source_names(joint_names=joint_names, joint_count=joint_count)

    if template == SKELETON_TEMPLATE_KEEP:
        seen: set[str] = set()
        return [_make_unique_name(name, seen) for name in source_names]

    template_names = _load_template_names(template)
    extra_prefix = _EXTRA_BONE_PREFIX[template]
    renamed: List[str] = []
    seen: set[str] = set()

    for idx in range(joint_count):
        if idx < len(template_names):
            candidate = template_names[idx]
        else:
            source_name = source_names[idx]
            if source_name and not _is_generic_bone_name(source_name):
                candidate = source_name
            else:
                candidate = f"{extra_prefix}_{idx - len(template_names):02d}"
        renamed.append(_make_unique_name(candidate, seen))
    return renamed


def apply_asset_joint_name_template(
    joint_names: Optional[Iterable[str]],
    joints: Optional[np.ndarray],
    parents: Optional[np.ndarray],
    template: Optional[str],
) -> List[str]:
    template = normalize_skeleton_template(template)
    joint_count = 0 if parents is None else int(len(parents))
    if joints is not None:
        joint_count = int(joints.shape[0])
    source_names = _resolve_source_names(joint_names=joint_names, joint_count=joint_count)

    if template == SKELETON_TEMPLATE_KEEP:
        return _make_unique_names(source_names)

    if joints is not None and parents is not None and joint_count > 0:
        humanoid_names = _apply_humanoid_template(
            source_names=source_names,
            joints=np.asarray(joints, dtype=np.float32),
            parents=np.asarray(parents, dtype=np.int32),
            template=template,
        )
        if humanoid_names is not None:
            return humanoid_names

    return apply_joint_name_template(
        joint_names=source_names,
        joint_count=joint_count,
        template=template,
    )
