import argparse
import json
import pathlib
import os
import multiprocessing as mp
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from tqdm import tqdm
from natsort import natsorted
from rich import print
import torch
import pickle

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import torch_utils
from general_motion_retargeting import IK_CONFIG_ROOT
import gc
import time
import psutil
import tracemalloc


def check_memory(threshold_gb=30):  # adjust based on your available memory
    mem = psutil.virtual_memory()
    used_memory_gb = (mem.total - mem.available) / (1024 ** 3)
    available_memory_gb = mem.available / (1024 ** 3)
    if available_memory_gb < threshold_gb:
        print(f"[WARNING] Memory usage:{used_memory_gb:.2f} GB, available:{available_memory_gb:.2f} GB, exceeding the threshold of {threshold_gb} GB.")
        return True
    return False


HERE = pathlib.Path(__file__).parent

SUPPORT_BODY_NAMES = {
    "unitree_g1": {
        "left": "left_ankle_roll_link",
        "right": "right_ankle_roll_link",
    },
    "unitree_g1_with_hands": {
        "left": "left_ankle_roll_link",
        "right": "right_ankle_roll_link",
    },
}

RETARGET_STABILIZER_CONFIG = {
    "passes": 2,
    "max_interp_gap": 6,
    "min_frames_after_drop": 30,
    "support_floor_margin": 0.06,
    "support_contact_margin": 0.04,
    "support_box_margin": 0.04,
    "support_violation_floor": 0.04,
    "root_xy_step_floor": 0.08,
    "root_z_step_floor": 0.05,
    "root_rot_step_floor": 0.45,
    "dof_step_floor": 0.45,
    "com_step_floor": 0.09,
    "step_sigma": 8.0,
}


def _parse_vec3(value, default="0 0 0"):
    return np.fromstring(value if value is not None else default, dtype=np.float32, sep=" ")


def _normalize_quat_np(quat):
    quat = np.asarray(quat, dtype=np.float32)
    quat_norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat_norm = np.clip(quat_norm, 1e-8, None)
    return quat / quat_norm


def _quat_angle_delta_xyzw(quat_a, quat_b):
    quat_a = _normalize_quat_np(quat_a)
    quat_b = _normalize_quat_np(quat_b)
    dot = np.abs(np.sum(quat_a * quat_b, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def _robust_upper_bound(values, min_threshold, sigma=6.0):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float(min_threshold)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_scale = 1.4826 * mad
    if robust_scale < 1e-6:
        return float(max(min_threshold, median * 3.0))
    return float(max(min_threshold, median + sigma * robust_scale))


def _frame_bad_mask_from_step_metric(step_metric, threshold):
    num_frames = len(step_metric)
    bad_mask = np.zeros(num_frames, dtype=bool)
    if num_frames == 0:
        return bad_mask
    if num_frames == 1:
        bad_mask[0] = step_metric[0] > threshold
        return bad_mask
    if num_frames == 2:
        bad_mask[:] = step_metric[1] > threshold
        return bad_mask

    bad_mask[1:-1] = (step_metric[1:-1] > threshold) & (step_metric[2:] > threshold)
    bad_mask[0] = (step_metric[1] > threshold) and (step_metric[2] > threshold)
    bad_mask[-1] = (step_metric[-1] > threshold) and (step_metric[-2] > threshold)
    return bad_mask


def _load_support_definition(xml_file, robot_name, kinematics_model, device):
    support_body_names = SUPPORT_BODY_NAMES.get(robot_name)
    if support_body_names is None:
        return None

    tree = ET.parse(xml_file)
    xml_doc_root = tree.getroot()
    xml_world_body = xml_doc_root.find("worldbody")
    if xml_world_body is None:
        return None

    target_body_names = set(support_body_names.values())
    body_points = {}

    def _walk_body(xml_body):
        body_name = xml_body.attrib.get("name")
        if body_name in target_body_names:
            points = []
            for geom_node in xml_body.findall("geom"):
                if geom_node.attrib.get("mesh") is not None:
                    continue
                if geom_node.attrib.get("type") == "mesh":
                    continue
                if "pos" not in geom_node.attrib:
                    continue
                points.append(_parse_vec3(geom_node.attrib["pos"]))
            if points:
                body_points[body_name] = np.stack(points, axis=0)
        for child_body in xml_body.findall("body"):
            _walk_body(child_body)

    for xml_body in xml_world_body.findall("body"):
        _walk_body(xml_body)

    support_definition = {}
    for side, body_name in support_body_names.items():
        body_points_side = body_points.get(body_name)
        if body_points_side is None or body_points_side.size == 0:
            continue
        try:
            body_idx = kinematics_model.get_body_idx(body_name)
        except ValueError:
            continue
        support_definition[side] = {
            "body_name": body_name,
            "body_idx": body_idx,
            "local_points": torch.from_numpy(body_points_side).to(device=device, dtype=torch.float),
        }

    return support_definition or None


def _compute_support_state(body_pos, body_rot, support_definition, config):
    num_frames = body_pos.shape[0]
    default_state = {
        "support_valid": np.zeros(num_frames, dtype=bool),
        "support_min_xy": np.zeros((num_frames, 2), dtype=np.float32),
        "support_max_xy": np.zeros((num_frames, 2), dtype=np.float32),
        "contact_mask": {},
    }
    if support_definition is None:
        return default_state

    foot_min_height = {}
    foot_points_xy = {}
    for side, entry in support_definition.items():
        body_idx = entry["body_idx"]
        local_points = entry["local_points"]
        num_points = local_points.shape[0]

        foot_pos = body_pos[:, body_idx, :]
        foot_rot = body_rot[:, body_idx, :]

        world_offset = torch_utils.quat_rotate(
            foot_rot.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 4),
            local_points.unsqueeze(0).expand(num_frames, -1, -1).reshape(-1, 3),
        ).reshape(num_frames, num_points, 3)
        world_points = foot_pos.unsqueeze(1) + world_offset

        foot_points_xy[side] = world_points[..., :2].detach().cpu().numpy()
        foot_min_height[side] = world_points[..., 2].amin(dim=1).detach().cpu().numpy()

    if not foot_min_height:
        return default_state

    stacked_min_height = np.stack(list(foot_min_height.values()), axis=1)
    sequence_ground = float(np.min(stacked_min_height))
    frame_ground = np.min(stacked_min_height, axis=1)

    contact_mask = {}
    for side, min_height in foot_min_height.items():
        contact_mask[side] = (
            (min_height <= frame_ground + config["support_contact_margin"])
            & (min_height <= sequence_ground + config["support_floor_margin"])
        )

    support_valid = np.zeros(num_frames, dtype=bool)
    support_min_xy = np.zeros((num_frames, 2), dtype=np.float32)
    support_max_xy = np.zeros((num_frames, 2), dtype=np.float32)
    for frame_idx in range(num_frames):
        contact_points = []
        for side, side_points_xy in foot_points_xy.items():
            if contact_mask[side][frame_idx]:
                contact_points.append(side_points_xy[frame_idx])
        if not contact_points:
            continue

        contact_points = np.concatenate(contact_points, axis=0)
        support_valid[frame_idx] = True
        support_min_xy[frame_idx] = contact_points.min(axis=0)
        support_max_xy[frame_idx] = contact_points.max(axis=0)

    return {
        "support_valid": support_valid,
        "support_min_xy": support_min_xy,
        "support_max_xy": support_max_xy,
        "contact_mask": contact_mask,
    }


def _evaluate_motion_sequence(root_pos, root_rot, dof_pos, kinematics_model, device, support_definition, config):
    with torch.no_grad():
        root_pos_tensor = torch.from_numpy(root_pos).to(device=device, dtype=torch.float)
        root_rot_tensor = torch.from_numpy(root_rot).to(device=device, dtype=torch.float)
        dof_pos_tensor = torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)

        body_pos, body_rot = kinematics_model.forward_kinematics(root_pos_tensor, root_rot_tensor, dof_pos_tensor)
        com_pos = kinematics_model.compute_center_of_mass_from_fk(body_pos, body_rot)
        support_state = _compute_support_state(body_pos, body_rot, support_definition, config)

    return {
        "com_pos": com_pos.detach().cpu().numpy(),
        "support_state": support_state,
    }


def _detect_bad_frames(root_pos, root_rot, dof_pos, eval_data, config):
    num_frames = root_pos.shape[0]
    com_pos = eval_data["com_pos"]
    support_state = eval_data["support_state"]

    root_xy_step = np.zeros(num_frames, dtype=np.float32)
    root_xy_step[1:] = np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1)

    root_z_step = np.zeros(num_frames, dtype=np.float32)
    root_z_step[1:] = np.abs(np.diff(root_pos[:, 2], axis=0))

    root_rot_step = np.zeros(num_frames, dtype=np.float32)
    root_rot_step[1:] = _quat_angle_delta_xyzw(root_rot[1:], root_rot[:-1])

    dof_step = np.zeros(num_frames, dtype=np.float32)
    if dof_pos.shape[1] > 0:
        dof_step[1:] = np.sqrt(np.mean(np.diff(dof_pos, axis=0) ** 2, axis=1))

    com_step = np.zeros(num_frames, dtype=np.float32)
    com_step[1:] = np.linalg.norm(np.diff(com_pos, axis=0), axis=1)

    support_excess = np.zeros(num_frames, dtype=np.float32)
    support_valid = support_state["support_valid"]
    if np.any(support_valid):
        support_min_xy = support_state["support_min_xy"][support_valid] - config["support_box_margin"]
        support_max_xy = support_state["support_max_xy"][support_valid] + config["support_box_margin"]
        support_com_xy = com_pos[support_valid, :2]

        dx = np.maximum.reduce(
            [
                support_min_xy[:, 0] - support_com_xy[:, 0],
                np.zeros_like(support_com_xy[:, 0]),
                support_com_xy[:, 0] - support_max_xy[:, 0],
            ]
        )
        dy = np.maximum.reduce(
            [
                support_min_xy[:, 1] - support_com_xy[:, 1],
                np.zeros_like(support_com_xy[:, 1]),
                support_com_xy[:, 1] - support_max_xy[:, 1],
            ]
        )
        support_excess[support_valid] = np.sqrt(dx ** 2 + dy ** 2)

    thresholds = {
        "support_excess": _robust_upper_bound(
            support_excess[support_valid], config["support_violation_floor"], sigma=config["step_sigma"]
        ),
        "root_xy_step": _robust_upper_bound(
            root_xy_step[1:], config["root_xy_step_floor"], sigma=config["step_sigma"]
        ),
        "root_z_step": _robust_upper_bound(
            root_z_step[1:], config["root_z_step_floor"], sigma=config["step_sigma"]
        ),
        "root_rot_step": _robust_upper_bound(
            root_rot_step[1:], config["root_rot_step_floor"], sigma=config["step_sigma"]
        ),
        "dof_step": _robust_upper_bound(dof_step[1:], config["dof_step_floor"], sigma=config["step_sigma"]),
        "com_step": _robust_upper_bound(com_step[1:], config["com_step_floor"], sigma=config["step_sigma"]),
    }

    reason_masks = {
        "support_excess": support_valid & (support_excess > thresholds["support_excess"]),
        "root_xy_step": _frame_bad_mask_from_step_metric(root_xy_step, thresholds["root_xy_step"]),
        "root_z_step": _frame_bad_mask_from_step_metric(root_z_step, thresholds["root_z_step"]),
        "root_rot_step": _frame_bad_mask_from_step_metric(root_rot_step, thresholds["root_rot_step"]),
        "dof_step": _frame_bad_mask_from_step_metric(dof_step, thresholds["dof_step"]),
        "com_step": _frame_bad_mask_from_step_metric(com_step, thresholds["com_step"]),
    }

    bad_mask = np.zeros(num_frames, dtype=bool)
    for reason_mask in reason_masks.values():
        bad_mask |= reason_mask

    metrics = {
        "support_excess": support_excess,
        "root_xy_step": root_xy_step,
        "root_z_step": root_z_step,
        "root_rot_step": root_rot_step,
        "dof_step": dof_step,
        "com_step": com_step,
        "support_valid": support_valid,
    }
    return bad_mask, reason_masks, metrics, thresholds


def _interpolate_root_rotations(quat_start, quat_end, num_frames):
    if num_frames <= 0:
        return np.zeros((0, 4), dtype=np.float32)
    key_rots = R.from_quat(np.stack([_normalize_quat_np(quat_start), _normalize_quat_np(quat_end)], axis=0))
    slerp = Slerp([0.0, 1.0], key_rots)
    interp_times = np.linspace(0.0, 1.0, num_frames + 2, dtype=np.float32)[1:-1]
    return slerp(interp_times).as_quat().astype(np.float32)


def _interpolate_bad_segments(root_pos, root_rot, dof_pos, bad_mask, max_interp_gap):
    corrected_mask = np.zeros_like(bad_mask)
    num_frames = len(bad_mask)
    seg_start = 0
    while seg_start < num_frames:
        if not bad_mask[seg_start]:
            seg_start += 1
            continue

        seg_end = seg_start
        while seg_end + 1 < num_frames and bad_mask[seg_end + 1]:
            seg_end += 1

        seg_len = seg_end - seg_start + 1
        left_idx = seg_start - 1
        right_idx = seg_end + 1
        if seg_len <= max_interp_gap:
            if left_idx >= 0 and right_idx < num_frames and not bad_mask[left_idx] and not bad_mask[right_idx]:
                alpha = np.linspace(0.0, 1.0, seg_len + 2, dtype=np.float32)[1:-1, None]
                root_pos[seg_start:seg_end + 1] = (1.0 - alpha) * root_pos[left_idx] + alpha * root_pos[right_idx]
                dof_pos[seg_start:seg_end + 1] = (1.0 - alpha) * dof_pos[left_idx] + alpha * dof_pos[right_idx]
                root_rot[seg_start:seg_end + 1] = _interpolate_root_rotations(
                    root_rot[left_idx], root_rot[right_idx], seg_len
                )
                corrected_mask[seg_start:seg_end + 1] = True
            elif left_idx < 0 and right_idx < num_frames and not bad_mask[right_idx]:
                root_pos[seg_start:seg_end + 1] = root_pos[right_idx]
                dof_pos[seg_start:seg_end + 1] = dof_pos[right_idx]
                root_rot[seg_start:seg_end + 1] = root_rot[right_idx]
                corrected_mask[seg_start:seg_end + 1] = True
            elif right_idx >= num_frames and left_idx >= 0 and not bad_mask[left_idx]:
                root_pos[seg_start:seg_end + 1] = root_pos[left_idx]
                dof_pos[seg_start:seg_end + 1] = dof_pos[left_idx]
                root_rot[seg_start:seg_end + 1] = root_rot[left_idx]
                corrected_mask[seg_start:seg_end + 1] = True

        seg_start = seg_end + 1

    return corrected_mask


def _summarize_reason_counts(reason_masks):
    return {reason: int(mask.sum()) for reason, mask in reason_masks.items() if int(mask.sum()) > 0}


def stabilize_retargeted_motion(root_pos, root_rot, dof_pos, kinematics_model, device, robot_name, xml_file):
    num_input_frames = root_pos.shape[0]
    stats = {
        "input_frames": int(num_input_frames),
        "support_tracking": False,
        "interpolated_frames": 0,
        "dropped_frames": 0,
        "initial_bad_frames": 0,
        "final_bad_frames": 0,
    }
    if num_input_frames < 3:
        stats["skipped_reason"] = "too_short"
        return root_pos, root_rot, dof_pos, stats

    support_definition = _load_support_definition(xml_file, robot_name, kinematics_model, device)
    stats["support_tracking"] = support_definition is not None

    root_pos = np.asarray(root_pos, dtype=np.float32).copy()
    root_rot = _normalize_quat_np(np.asarray(root_rot, dtype=np.float32).copy())
    dof_pos = np.asarray(dof_pos, dtype=np.float32).copy()

    total_interpolated_mask = np.zeros(num_input_frames, dtype=bool)
    initial_metrics = None
    initial_reason_masks = None
    final_metrics = None
    final_reason_masks = None

    for _ in range(RETARGET_STABILIZER_CONFIG["passes"] + 1):
        eval_data = _evaluate_motion_sequence(
            root_pos,
            root_rot,
            dof_pos,
            kinematics_model,
            device,
            support_definition,
            RETARGET_STABILIZER_CONFIG,
        )
        bad_mask, reason_masks, metrics, _ = _detect_bad_frames(
            root_pos, root_rot, dof_pos, eval_data, RETARGET_STABILIZER_CONFIG
        )

        if initial_metrics is None:
            initial_metrics = metrics
            initial_reason_masks = reason_masks
            stats["initial_bad_frames"] = int(bad_mask.sum())
            stats["initial_bad_reason_counts"] = _summarize_reason_counts(reason_masks)

        if not np.any(bad_mask):
            final_metrics = metrics
            final_reason_masks = reason_masks
            break

        corrected_mask = _interpolate_bad_segments(
            root_pos,
            root_rot,
            dof_pos,
            bad_mask,
            RETARGET_STABILIZER_CONFIG["max_interp_gap"],
        )
        total_interpolated_mask |= corrected_mask
        if not np.any(corrected_mask):
            final_metrics = metrics
            final_reason_masks = reason_masks
            break
    else:
        final_metrics = metrics
        final_reason_masks = reason_masks

    if final_metrics is None or final_reason_masks is None:
        eval_data = _evaluate_motion_sequence(
            root_pos,
            root_rot,
            dof_pos,
            kinematics_model,
            device,
            support_definition,
            RETARGET_STABILIZER_CONFIG,
        )
        _, final_reason_masks, final_metrics, _ = _detect_bad_frames(
            root_pos, root_rot, dof_pos, eval_data, RETARGET_STABILIZER_CONFIG
        )

    stats["interpolated_frames"] = int(total_interpolated_mask.sum())

    final_bad_mask = np.zeros(root_pos.shape[0], dtype=bool)
    for reason_mask in final_reason_masks.values():
        final_bad_mask |= reason_mask

    dropped_mask = np.zeros_like(final_bad_mask)
    if np.any(final_bad_mask):
        keep_mask = ~final_bad_mask
        if int(keep_mask.sum()) >= RETARGET_STABILIZER_CONFIG["min_frames_after_drop"]:
            dropped_mask = final_bad_mask.copy()
            root_pos = root_pos[keep_mask]
            root_rot = root_rot[keep_mask]
            dof_pos = dof_pos[keep_mask]

            eval_data = _evaluate_motion_sequence(
                root_pos,
                root_rot,
                dof_pos,
                kinematics_model,
                device,
                support_definition,
                RETARGET_STABILIZER_CONFIG,
            )
            _, final_reason_masks, final_metrics, _ = _detect_bad_frames(
                root_pos, root_rot, dof_pos, eval_data, RETARGET_STABILIZER_CONFIG
            )

    final_bad_mask = np.zeros(root_pos.shape[0], dtype=bool)
    for reason_mask in final_reason_masks.values():
        final_bad_mask |= reason_mask

    stats["dropped_frames"] = int(dropped_mask.sum())
    stats["output_frames"] = int(root_pos.shape[0])
    stats["final_bad_frames"] = int(final_bad_mask.sum())
    stats["final_bad_reason_counts"] = _summarize_reason_counts(final_reason_masks)

    if initial_metrics is not None and np.any(initial_metrics["support_valid"]):
        initial_support_excess = initial_metrics["support_excess"][initial_metrics["support_valid"]]
        stats["support_excess_mean_before"] = float(np.mean(initial_support_excess))
        stats["support_excess_p95_before"] = float(np.percentile(initial_support_excess, 95))
    if final_metrics is not None and np.any(final_metrics["support_valid"]):
        final_support_excess = final_metrics["support_excess"][final_metrics["support_valid"]]
        stats["support_excess_mean_after"] = float(np.mean(final_support_excess))
        stats["support_excess_p95_after"] = float(np.percentile(final_support_excess, 95))

    return root_pos, root_rot, dof_pos, stats


def process_file(smplx_file_path, tgt_file_path, tgt_robot, SMPLX_FOLDER, tgt_folder, total_files, verbose=False):
    def log_memory(message):
        if verbose:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 ** 3)  # Convert to GB
            print(f"[MEMORY] {message}: {memory_usage:.2f} GB")
    
    # Start memory tracking if verbose
    if verbose:
        tracemalloc.start()
        
    # Initial checks (with optional logging)
    log_memory("Initial memory usage")
    
    num_pause = 0
    while check_memory():
        print(f"[PAUSE] Paused processing {smplx_file_path} to prevent memory overflow. num_pause: {num_pause}")
        time.sleep(60*2)
        num_pause += 1
        if num_pause > 10:
            print(f"[ERROR] Memory usage is still high after 10 pauses. Exiting.")
            return

    try:
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(smplx_file_path, SMPLX_FOLDER)
        mocap_frame_rate = smplx_data["mocap_frame_rate"]
        log_memory("After loading SMPL-X data")
    except Exception as e:
        print(f"Error loading {smplx_file_path}: {e}")
        return
    
  
    tgt_fps = 30
    try:
        smplx_frame_data_list, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        return
    
    # retarget
    retargeter = GMR(
        src_human="smplx",
        tgt_robot=tgt_robot,
        actual_human_height=actual_human_height,
    )
    qpos_list = []
    for smplx_frame_data in smplx_frame_data_list:
        qpos = retargeter.retarget(smplx_frame_data)
        qpos_list.append(qpos.copy())

    qpos_list = np.array(qpos_list)

    log_memory("After retargeting")
    
    device = os.environ.get("GMR_DEVICE", "cpu")
    kinematics_model = KinematicsModel(retargeter.xml_file, device=device)

    try:
        root_pos = qpos_list[:, :3]
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        return
    root_rot = qpos_list[:, 3:7]
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
    dof_pos = qpos_list[:, 7:]
    root_pos, root_rot, dof_pos, stabilization_stats = stabilize_retargeted_motion(
        root_pos,
        root_rot,
        dof_pos,
        kinematics_model,
        device,
        tgt_robot,
        retargeter.xml_file,
    )
    num_frames = root_pos.shape[0]

    stabilization_summary = (
        f"[GMR] Stabilized {os.path.basename(smplx_file_path)}: "
        f"bad {stabilization_stats['initial_bad_frames']} -> {stabilization_stats['final_bad_frames']}, "
        f"interp {stabilization_stats['interpolated_frames']}, "
        f"drop {stabilization_stats['dropped_frames']}"
    )
    if "support_excess_mean_before" in stabilization_stats and "support_excess_mean_after" in stabilization_stats:
        stabilization_summary += (
            f", support {stabilization_stats['support_excess_mean_before']:.4f}"
            f" -> {stabilization_stats['support_excess_mean_after']:.4f} m"
        )
    print(stabilization_summary)

    fk_root_pos = torch.zeros((num_frames, 3), device=device)
    fk_root_rot = torch.zeros((num_frames, 4), device=device)
    fk_root_rot[:, -1] = 1.0

    local_body_pos, _ = kinematics_model.forward_kinematics(
        fk_root_pos, fk_root_rot, torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )

    log_memory("After forward kinematics")

    body_names = kinematics_model.body_names
    
    HEIGHT_ADJUST = True
    if HEIGHT_ADJUST:
        # height adjust to ensure the lowerset part is on the ground
        body_pos, _ = kinematics_model.forward_kinematics(torch.from_numpy(root_pos).to(device=device, dtype=torch.float), 
                                                        torch.from_numpy(root_rot).to(device=device, dtype=torch.float), 
                                                        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)) # TxNx3
        ground_offset = 0.0
        lowerst_height = torch.min(body_pos[..., 2]).item()
        root_pos[:, 2] = root_pos[:, 2] - lowerst_height + ground_offset # make sure motion on the ground
        
    ROOT_ORIGIN_OFFSET = True
    if ROOT_ORIGIN_OFFSET:
        # offset using the first frame
        root_pos[:, :2] -= root_pos[0, :2]
        
        
    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": body_names,
        "retarget_stabilization": stabilization_stats,
    }


    os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
    with open(tgt_file_path, "wb") as f:
        pickle.dump(motion_data, f)
        
    # Progress print based on tgt_folder
    done = 0
    for root, _, files in os.walk(tgt_folder):
        done += len([f for f in files if f.endswith('.pkl')])
    print(f"Processed {done}/{total_files}: {tgt_file_path}")
    
    if verbose:
        # Get memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        print("\nTop 10 memory-consuming lines:")
        for stat in top_stats[:10]:
            print(stat)
        
        tracemalloc.stop()
        
    # clean cache
    torch.cuda.empty_cache()
    gc.collect()
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--src_folder", type=str,
                        required=True,
                        )
    parser.add_argument("--tgt_folder", type=str,
                        required=True,
                        )
    
    parser.add_argument("--override", default=False, action="store_true")
    parser.add_argument("--num_cpus", default=4, type=int)
    args = parser.parse_args()
    
    # print the total number of cpus and gpus
    print(f"Total CPUs: {mp.cpu_count()}")
    print(f"Using {args.num_cpus} CPUs.")
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    hard_motions_folder = HERE / ".." / "assets" / "hard_motions"

    verbose = False

    hard_motions_paths = [hard_motions_folder / "0.txt", 
                          hard_motions_folder / "1.txt"]
    hard_motions = []
    for hard_motions_path in hard_motions_paths:
        with open(hard_motions_path, "r") as f:
            for line in f:
                if "Motion:" in line:
                    motion_path = line.split(":")[1].strip()
                else:
                    continue
                motion_path = motion_path.split(",")[0].strip().split(".")[0]
                hard_motions.append(motion_path)
                
                
    args_list = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in natsorted(filenames):
            if filename.endswith("_stagei.npz"):
                continue
            if filename.endswith((".pkl", ".npz")):
                smplx_file_path = os.path.join(dirpath, filename)
                tgt_file_path = smplx_file_path.replace(src_folder, tgt_folder).replace(".npz", ".pkl")
                if not os.path.exists(tgt_file_path) or args.override:
                    args_list.append((smplx_file_path, tgt_file_path, args.robot, SMPLX_FOLDER, tgt_folder))
    print("full args_list:", len(args_list))
    
    # remove hard and infeasible motions
    exclude_file_content = ["BMLrub", "EKUT", "crawl", "_lie", "upstairs", "downstairs"]
    
    new_args_list = []
    for arguments in args_list:
        motion_name = arguments[0].split("/")[-1].split('.')[0]
        if motion_name in hard_motions:
            continue
        if any(content in motion_name for content in exclude_file_content):
            continue
        new_args_list.append(arguments)
    args_list = new_args_list
    
    
    print("new args_list:", len(args_list))
    
    total_files = len(args_list)
    print(f"Total number of files to process: {total_files}")
    with mp.Pool(args.num_cpus) as pool:
        pool.starmap(process_file, [args + (total_files, verbose) for args in args_list])

    print("Done. Saved to ", tgt_folder)


if __name__ == "__main__":
    main()
