INPUT_MODEL_SIZE_VIT = (616, 1064)
IMAGE_NET_MEAN = [123.675, 116.28, 103.53]
IMAGE_NET_STD = [58.395, 57.12, 57.375]
# padding to input_size
IMG_PADDING = [123.675, 116.28, 103.53]
CANONICAL_FX = 1000.

KEY_INTRINSICS = 'intrinsics'
KEY_POSE_FRAME_TO_WORLD = 'pose_frame_to_world'
KEY_RGB = 'rgb'
KEY_GT_DEPTH = 'gt_depth'
KEY_FOLDER_POSE = 'pose'
KEY_FOLDER_DEPTH = 'depth'
KEY_FOLDER_IMAGE = 'color'
KEY_FRAMES = 'frames'

# constants for mesh fusing
VOXEL_SIZE = 0.05
TSDF_CUBIC_SIZE = 3.0 / 512.
SDF_TRUNC = 0.04
MAX_DEPTH = 7.
