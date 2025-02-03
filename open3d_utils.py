from typing import Union
import open3d as o3d
import numpy as np
import os

from constants import MAX_DEPTH


def visualize_camera_poses(poses: Union[list, np.ndarray], pointcloud: np.ndarray):
    """
    Visualize camera poses and 3D point cloud

    :param: poses (list of np.array): List of 4 x 4 camera-to-world poses
    :pram pointcloud: N x 3 point cloud array
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add point cloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pointcloud)
    vis.add_geometry(pc)

    # Add camera frustums
    for pose in poses:
        cam = o3d.geometry.LineSet()
        points = [
            [0, 0, 0],
            [0.1, 0.1, 0.2],
            [-0.1, 0.1, 0.2],
            [-0.1, -0.1, 0.2],
            [0.1, -0.1, 0.2]
        ]
        points = np.array(points)
        points = (pose[:3, :3] @ points.T).T + pose[:3, 3]
        cam.points = o3d.utility.Vector3dVector(points)
        cam.lines = o3d.utility.Vector2iVector([
            [0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]
        ])
        vis.add_geometry(cam)

    vis.run()
    vis.destroy_window()


def add_o3d_images(img: np.ndarray, depth: np.ndarray):
    """
    Convert image and depth to o3d Image format and create RGBD o3d image

    :param img_orig: image
    :param depth: depth map 
    """
    color_o3d = o3d.geometry.Image(img)
    depth_o3d = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, depth_trunc=MAX_DEPTH, depth_scale=1., convert_rgb_to_intensity=False)

    return rgbd


def save_pointcloud_o3d(p3d: list, colors: list, scene_name: str, path_to_scene: str, use_gt_depth: bool = False, save_pointcloud: bool = True):
    """
    Save poincloud with colors to .ply

    :param p3d: list of points
    :param colors: list of rgb colors for points
    :param scene_name: name of the scene
    :parm path_to_scene: path to scene
    :param use_gt_depth: whether gt depth was used
    :param save_pointcloud: whether to save pointcloud
    """
    if save_pointcloud:
        p3d = np.vstack(p3d)
        colors = np.vstack(colors)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(p3d)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        file_name = os.path.join(path_to_scene, f'gt_depth_{scene_name}_pc.ply') if use_gt_depth else os.path.join(path_to_scene, f'pred_depth_{scene_name}_pc.ply')
        o3d.io.write_point_cloud(file_name, point_cloud)
