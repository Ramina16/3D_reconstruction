from typing import Tuple
import numpy as np


def points_2d_to_3d(depth: np.array, intrinsics: np.array, step: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 2D points to 3D in camera space

    :param depth: depth map
    :param intrinsics: intrinsics matrix
    :param step: step with which pointcloud will be proceed
    :return: N x 3 pointcloud, mask
    """
    h, w = depth.shape[:2]
    depth_flatten = depth.flatten()
    mask = (depth_flatten > 0.).astype(float)
    depth_flatten = depth_flatten[mask > 0.]
    
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    uv_h = np.stack((u, v, np.ones_like(u)), axis=-1).reshape(-1, 3)

    # Back-project to camera coordinates
    K_inv = np.linalg.inv(intrinsics)
    points_camera = (K_inv @ uv_h.T).T
    points_camera = points_camera[mask > 0.]
    points_camera *= depth_flatten[:, None]

    return points_camera[::step], mask


def points_3d_to_world(pointcloud: np.ndarray, pose_frame_to_world: np.ndarray) -> np.ndarray:
    """
    Project 3D pointloud to world coordinates

    :param pointcloud: N x 3 pointcloud in camera coordinate system
    :param pose_frame_to_world: 4 x 4 matrix of pose frame_to_world
    :return: N x 3 pointcloud in world coordinates
    """
    points_h = np.hstack((pointcloud, np.ones((pointcloud.shape[0], 1))))
    points_world = (pose_frame_to_world @ points_h.T).T

    return points_world[:, :3]


def project_add_points_color(save_pointcloud: bool, depth: np.ndarray, intrinsics: np.ndarray, step: int, pose_frame_to_world: np.ndarray,
                             p3d: list, img: np.ndarray, colors: list) -> np.ndarray:
    """
    Project points to 3D in world coordinates for current frame and add to previous points

    :param save_pointcloud: whether to project and save points
    :param depth: depth map
    :param intrinsics: intrinsic matrix
    :param step: step with which pointcloud will be proceed
    :param pose_frame_to_world: 4 x 4 matrix of pose frame_to_world
    :param p3d: list storing previously accumulated 3D points in world coordinates
    :param img: RGB image
    :param colors: list storing previously accumulated color values corresponding to 3D points

    :return: updated lists of 3D points and corresponding colors
    """
    if save_pointcloud:
        pointcloud_c, mask = points_2d_to_3d(depth, intrinsics, step=step)
        pointcloud_w = points_3d_to_world(pointcloud_c, pose_frame_to_world)

        p3d.append(pointcloud_w)

        colors_valid = img.reshape(-1, 3) / 255.0
        colors.append(colors_valid[mask > 0.][::step])
    
    return p3d, colors
