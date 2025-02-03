import numpy as np
import os
import torch
import open3d as o3d
from pointcloud_utils import project_add_points_color
from scannet_scene_utils import load_scene
from utils import create_parser, load_model, preprocess_img_for_metric3d, postprocess_depth_metric3d
from constants import KEY_INTRINSICS, KEY_POSE_FRAME_TO_WORLD, KEY_GT_DEPTH, KEY_RGB, KEY_FRAMES, SDF_TRUNC, TSDF_CUBIC_SIZE
from open3d_utils import add_o3d_images, save_pointcloud_o3d



if __name__ == '__main__':
    args = create_parser()

    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

    scannet_scene = load_scene(args.path_to_scene)
    scene_name = os.path.basename(args.path_to_scene)

    model, scale = load_model(args.model_name, device=device)

    intrinsics = scannet_scene[KEY_INTRINSICS]
    intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(round(intrinsics[0, 2]) * 2, round(intrinsics[1, 2]) * 2, intrinsics[0, 0], intrinsics[1, 1], 
                                                       intrinsics[0, 2], intrinsics[1, 2])
    # needed for depth postprocessing after metric3d model prediction
    intrinsics_sc = [intrinsics[0, 0] * scale, intrinsics[1, 1] * scale, intrinsics[0, 2] * scale, intrinsics[1, 2] * scale]
    

    if args.save_pointcloud:
        p3d = []
        colors = []
    count = 0
    use_gt = args.use_gt_depth

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=TSDF_CUBIC_SIZE,
        sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for frame in scannet_scene[KEY_FRAMES]:
        if count % 5 != 0:
            count += 1
            continue
        
        print(f'Processing {count} frame')

        pose_frame_to_world = frame[KEY_POSE_FRAME_TO_WORLD]
        img_orig = frame[KEY_RGB]

        if use_gt:
            depth = frame[KEY_GT_DEPTH]
        else:
            img, pad_info = preprocess_img_for_metric3d(img_orig, scale, device)
            with torch.no_grad():
                depth, confidence, output_dict = model.inference({'input': img})
        
            depth = postprocess_depth_metric3d(depth, img_orig, pad_info, intrinsics_sc)
        
        rgbd = add_o3d_images(img_orig, depth)
        volume.integrate(rgbd, intrinsics_o3d, np.linalg.inv(pose_frame_to_world))

        p3d, colors = project_add_points_color(args.save_pointcloud, depth, intrinsics, args.step, pose_frame_to_world, p3d, img_orig, colors)

        count += 1

    mesh = volume.extract_triangle_mesh()

    save_pointcloud_o3d(p3d, colors, scene_name, args.path_to_scene, args.use_gt_depth, args.save_pointcloud)

    mesh_name = f'gt_depth_{scene_name}_mesh.ply' if args.use_gt_depth else f'pred_depth_{scene_name}_mesh.ply'
    o3d.io.write_triangle_mesh(os.path.join(args.path_to_scene, mesh_name), mesh)
