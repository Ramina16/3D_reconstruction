import numpy as np
import cv2
import os

from constants import KEY_FOLDER_IMAGE, KEY_FOLDER_DEPTH, KEY_FOLDER_POSE, KEY_FRAMES, KEY_POSE_FRAME_TO_WORLD, KEY_INTRINSICS, KEY_GT_DEPTH, KEY_RGB


def load_intrinsics(file_path: str) -> np.ndarray:
    """
    Load intrinsics matrix from a text file

    :param file_path: path to .txt file with intrinsics
    :return: 3x3 intrinsics matrix
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        intrinsics = np.array([[float(x) for x in line.strip().split()] for line in lines])

    return intrinsics[:3, :3]


def load_pose(file_path: str) -> np.ndarray:
    """
    Load a 4x4 pose matrix from a text file. Each pose file contains:
    r11 r12 r13 tx
    r21 r22 r23 ty
    r31 r32 r33 tz
    0   0   0   1

    :param file_path: path to .txt file with poses
    :return: 4x4 pose frame_to_world
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        pose = np.array([[float(x) for x in line.strip().split()] for line in lines])

    return pose


def load_depth(file_path: str, scale: float = 1000.0) -> np.ndarray:
    """
    Load depth map from a 16-bit PNG file and convert to meters

    :param file_path: path to .png depth file
    :param scale: scale parameter to convert depth to meters
    :return: np.array of depth in meters
    """
    depth = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to load depth map from {file_path}")
    
    return depth.astype(np.float32) / scale


def load_rgb(file_path: str):
    """
    Load RGB image from file

    :param file_path: str
    """
    rgb = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError(f"Failed to load RGB image from {file_path}")
    
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def load_scene(scene_folder: str) -> dict:
    """
    Load all images, depth maps, intrinsics, and poses from a ScanNet scene folder
    """
    intrinsics = load_intrinsics(os.path.join(scene_folder, "intrinsics.txt"))

    pose_folder = os.path.join(scene_folder, KEY_FOLDER_POSE)
    depth_folder = os.path.join(scene_folder, KEY_FOLDER_DEPTH)
    rgb_folder = os.path.join(scene_folder, KEY_FOLDER_IMAGE)

    pose_files = sorted(os.listdir(pose_folder))
    depth_files = sorted(os.listdir(depth_folder))
    rgb_files = sorted(os.listdir(rgb_folder))

    data = {}
    data[KEY_INTRINSICS] = intrinsics
    data[KEY_FRAMES] = []
    for pose_file, depth_file, rgb_file in zip(pose_files, depth_files, rgb_files):
        pose = load_pose(os.path.join(pose_folder, pose_file))
        depth = load_depth(os.path.join(depth_folder, depth_file))
        rgb = load_rgb(os.path.join(rgb_folder, rgb_file))

        data[KEY_FRAMES].append({
            KEY_POSE_FRAME_TO_WORLD: pose,
            KEY_GT_DEPTH: depth,
            KEY_RGB: rgb,
        })

    return data
