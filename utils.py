import argparse
from typing import Tuple
import numpy as np
import torch
import cv2

from constants import INPUT_MODEL_SIZE_VIT, IMAGE_NET_MEAN, IMAGE_NET_STD, IMG_PADDING, CANONICAL_FX


def create_parser():
    """
    Create parser for input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, default='metric3d_vit_small', help='model name, can be one of \
                        [metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2], see https://github.com/YvanYin/Metric3D')
    parser.add_argument('--step', '-step', type=int, default=1, help='points processing step')
    parser.add_argument('--path_to_scene', '-path_to_scene', type=str, help='path to folder with scannet scene')
    parser.add_argument('--use_gt_depth', '-use_gt_depth', action='store_true', help='whether to use GT depth instead of predicted by model')
    parser.add_argument('--save_pointcloud', '-save_pc', action='store_true', help='whether to save pointcloud in addition to mesh')

    args = parser.parse_args()

    return args



def load_model(model_name: str = 'metric3d_vit_small', device: str = 'cpu', h_img: int = 240, w_img: int = 320):
    """
    Load model and calculate scale between input model size and input image

    :param model_name: name of the model
    :param device: device to be used for model inference (cpu or gpu)
    :param h_img: height of the image
    :param w_img: width of the image

    :return: model instance, scale
    """
    scale = min(INPUT_MODEL_SIZE_VIT[0] / h_img, INPUT_MODEL_SIZE_VIT[1] / w_img)
    model = torch.hub.load('yvanyin/metric3d', model_name, pretrain=True)
    model.to(device)
    model.eval()

    return model, scale


def preprocess_img_for_metric3d(img_orig: np.ndarray, scale: float, device: str = 'cpu') -> Tuple[torch.tensor, list]:
    """
    Image preprocessing for metric3d model, see https://github.com/YvanYin/Metric3D

    :param img_orig: original image
    :param scale: scale between input model size and input image
    :param device: device to be used (cpu or gpu)
    """
    h, w = img_orig.shape[:2]
    img = cv2.resize(img_orig, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    pad_h = INPUT_MODEL_SIZE_VIT[0] - h
    pad_w = INPUT_MODEL_SIZE_VIT[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2

    img = cv2.copyMakeBorder(img, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=IMG_PADDING)

    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    mean = torch.tensor(IMAGE_NET_MEAN).float()[:, None, None]
    std = torch.tensor(IMAGE_NET_STD).float()[:, None, None]
    img = torch.from_numpy(img.transpose((2, 0, 1))).float()
    img = torch.div((img - mean), std)
    img = img[None, :, :, :].to(device)

    return img, pad_info


def postprocess_depth_metric3d(depth_orig: np.ndarray, img_orig: np.ndarray, pad_info: np.ndarray, intrinsics_sc: list, max_depth: float = 50.) -> np.ndarray:
    """
    Postprocessing for otput depth from model

    :param depth_orig: predicted depth
    :param img_orig: original image
    :param pad_info: padding info that was added to iriginal image
    :param intrinsics_sc: list of scaled fx, fy, cx, cy
    :param max_depth: max depth to be proceed
    :return: modified depth map
    """
    pred_depth = depth_orig.squeeze()
    pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] - pad_info[1], pad_info[2]: pred_depth.shape[1] - pad_info[3]]
  
    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], img_orig.shape[:2], mode='bilinear').squeeze()

    #### de-canonical transform (convert to metric depth)
    canonical_to_real_scale = intrinsics_sc[0] / CANONICAL_FX
    pred_depth = pred_depth * canonical_to_real_scale
    pred_depth = torch.clamp(pred_depth, 0, max_depth)

    pred_depth = pred_depth.cpu().numpy()

    return pred_depth
