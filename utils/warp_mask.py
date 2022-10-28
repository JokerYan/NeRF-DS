import os
import cv2
import sys
import json
import gin
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from absl import logging
from tqdm import tqdm

from hypernerf import configs
from hypernerf import models
from hypernerf import utils
from hypernerf import datasets

if os.path.exists(os.path.expanduser('~/hypernerf-barf/')):
    project_root = os.path.expanduser('~/hypernerf-barf/')
elif os.path.exists(os.path.expanduser('~/3d_cv/repos/hypernerf_barf/')):
    project_root = os.path.expanduser('~/3d_cv/repos/hypernerf_barf/')
else:
    raise NotImplemented
sys.path.insert(0, project_root)

if os.path.exists('/ssd/zhiwen/data/hypernerf/raw/'):
    data_root = '/ssd/zhiwen/data/hypernerf/raw/'
elif os.path.exists('/hdd/zhiwen/data/hypernerf/raw/'):
    data_root = '/hdd/zhiwen/data/hypernerf/raw/'
elif os.path.exists('/home/zwyan/3d_cv/data/hypernerf/raw/'):
    data_root = '/home/zwyan/3d_cv/data/hypernerf/raw/'
else:
    raise NotImplemented

# dataset = '011_bell_07_novel_view'
# exp_name = '011_b07_nv_ms_exp01'
dataset = '015_cup_02_novel_view'
exp_name = '015_c02_nv_ms_exp22'
train_dir = os.path.join(project_root, 'experiments', exp_name)
data_dir = os.path.join(data_root, dataset)

depth_max = 2.0
depth_step = 0.01

def load_checkpoint():
  checkpoint_dir = Path(train_dir, 'checkpoints')
  checkpoint_dir.mkdir(exist_ok=True, parents=True)

  config_path = Path(train_dir, 'config.gin')
  with open(config_path, 'r') as f:
    logging.info('Loading config from %s', config_path)
    config_str = f.read()
  gin.parse_config(config_str)

  config_path = Path(train_dir, 'config.gin')
  with open(config_path, 'w') as f:
    logging.info('Saving config to %s', config_path)
    f.write(config_str)

  exp_config = configs.ExperimentConfig()
  train_config = configs.TrainConfig()
  eval_config = configs.EvalConfig()
  spec_config = configs.SpecularConfig()

  if spec_config.use_hyper_spec_model:
    dummy_model = models.HyperSpecModel({}, 0, 0)
  else:
    dummy_model = models.NerfModel({}, 0, 0)

  datasource = exp_config.datasource_cls(
    data_dir=data_dir,
    image_scale=exp_config.image_scale,
    random_seed=exp_config.random_seed,
    # Enable metadata based on model needs.
    use_warp_id=dummy_model.use_warp,
    use_appearance_id=(
            dummy_model.nerf_embed_key == 'appearance'
            or dummy_model.hyper_embed_key == 'appearance'),
    use_camera_id=dummy_model.nerf_embed_key == 'camera',
    use_time=dummy_model.warp_embed_key == 'time')

  config_dict = {
    'exp_config': exp_config,
    'train_config': train_config,
    'eval_config': eval_config,
    'spec_config': spec_config
  }
  return datasource, config_dict


def load_cameras(datasource, camera_name):
  camera_dir = Path(data_dir, camera_name)
  print(f'Loading cameras from {camera_dir}')
  camera_paths = datasource.glob_cameras(camera_dir)
  cameras = utils.parallel_map(datasource.load_camera, camera_paths, show_pbar=True)
  return cameras, camera_paths


def load_camera_pairs_with_masks(datasource, config_dict):
  train_cameras, train_camera_paths = load_cameras(datasource, 'train_camera')
  test_cameras, test_camera_paths = load_cameras(datasource, 'vrig_camera')
  train_masks = load_camera_masks(train_camera_paths, config_dict)
  test_masks = load_camera_masks(test_camera_paths, config_dict)
  assert len(train_cameras) == len(test_cameras)
  camera_pairs = []
  mask_pairs = []
  for i in range(len(train_cameras)):
    camera_pairs.append((train_cameras[i], test_cameras[i]))
    mask_pairs.append((train_masks[i], test_masks[i]))
  return camera_pairs, mask_pairs


def load_camera_masks(camera_paths, config_dict):
  exp_config = config_dict['exp_config']
  mask_dir = Path(data_dir, 'resized_mask', f"{int(exp_config.image_scale)}x")
  print(f"Loading masks from {mask_dir}")
  mask_list = datasets.load_camera_masks(mask_dir, camera_paths, exp_config.image_scale)
  return mask_list


def load_depth_map_list():
  depth_map_path = Path(train_dir, 'render_depth_train_camera')
  depth_map_list = np.load(str(depth_map_path))
  return depth_map_list


def cal_frame_step(depth_map_list, camera_pairs):
  render_count = len(depth_map_list)
  camera_count = len(camera_pairs)
  frame_step = camera_count // render_count
  assert frame_step == 1 or frame_step == 9
  return frame_step


def grid_sample_depth_map(image,    # H x W
                          pixels,   # H x W x N x 2
                          ):
  # pixels = np.concatenate([pixels[..., 1][..., np.newaxis],
  #                          pixels[..., 0][..., np.newaxis]], axis=-1)   # x,y   H x W x N x 2
  pixels = pixels.transpose([2, 0, 1, 3])       # N x H x W x 2
  N, H, W, _ = pixels.shape
  image = image[np.newaxis, np.newaxis, ...]    # 1 x C x H x W
  image = np.broadcast_to(image, [N, 1, H, W])  # N x C x H x W

  size_array = np.array([W, H])[np.newaxis, np.newaxis, np.newaxis, :]    # 1 x 1 x 1 x 2
  pixels = (pixels / size_array) * 2 - 1    # normalize

  image = torch.Tensor(image)
  pixels = torch.Tensor(pixels)
  sampled_depth = F.grid_sample(image, pixels, padding_mode='zeros')    # N x 1 x H x W
  sampled_depth = sampled_depth.numpy()
  sampled_depth = sampled_depth.transpose([2, 3, 0, 1]).squeeze(-1)     # H x W x D
  return sampled_depth


def sample_test_points(test_camera):
  pixels = test_camera.get_pixel_centers()
  height, width, _ = pixels.shape

  depth = np.arange(0, depth_max, depth_step)
  depth_count = len(depth)

  # reshape
  depth = np.broadcast_to(depth[np.newaxis, np.newaxis, :], [height, width, depth_count])
  pixels = np.broadcast_to(pixels[:, :, np.newaxis, :], [height, width, depth_count, 2])

  points = test_camera.pixels_to_points(pixels, depth)
  return points, depth


def get_in_range_mask_from_pixels(pixels):
  height, width, depth_count, _ = pixels.shape
  batch_shape = pixels.shape[:-1]
  pixels_flat = pixels.reshape([-1, 2])
  in_range_mask = (pixels_flat[:, 0] >= 0) & (pixels_flat[:, 0] < width) \
                  & (pixels_flat[:, 1] >= 0) & (pixels_flat[:, 1] < height)
  in_range_mask = in_range_mask.reshape(batch_shape)
  return in_range_mask


def get_depth_for_pixels(train_depth_map, pixels, in_range_mask):
  out_range_mask = np.logical_not(in_range_mask)
  out_range_mask = np.broadcast_to(out_range_mask[..., np.newaxis], pixels.shape)   # H x W x D x 2

  # batch_shape = pixels.shape[:-1]
  pixels[out_range_mask] = 0
  # pixels = pixels.reshape([-1, 2])
  # pixel_depths = train_depth_map[pixels[..., 1], pixels[..., 0]]  # y,x

  pixel_depths = grid_sample_depth_map(train_depth_map, pixels)  # H x W x D
  # pixel_depths = pixel_depths.reshape(batch_shape)
  return pixel_depths


def find_best_projected_points(test_points, train_camera, train_depth_map):
  pixels = train_camera.project(test_points)  # H x W x D x 2, in the form of [x, y]
  # pixels = np.array(np.round(pixels), np.int32)
  in_range_mask = get_in_range_mask_from_pixels(pixels)   # H x W x D
  pixel_depths = get_depth_for_pixels(train_depth_map, pixels, in_range_mask)     # H x W x D

  local_points = train_camera.points_to_local_points(test_points)
  point_depths = local_points[:, :, :, -1]    # H x W x D

  # find best point
  # depth_diff = np.abs(pixel_depths - point_depths)
  depth_diff = np.abs(1 / pixel_depths - 1 / point_depths)
  out_range_mask = np.logical_not(in_range_mask)
  depth_diff[out_range_mask] = float('inf')

  best_depth_idx = np.argmin(depth_diff, axis=-1)   # H x W
  best_depth_diff = np.min(depth_diff, axis=-1)     # H x W
  no_match_mask = best_depth_diff == float('inf')   # H x W
  best_depth_idx[no_match_mask] = 0

  best_pixels = np.take_along_axis(pixels, best_depth_idx[..., np.newaxis, np.newaxis], axis=-2)
  best_pixels = best_pixels.squeeze(-2)

  return best_depth_idx, best_pixels, no_match_mask


def get_mask_from_pixels(mask, pixels, no_match_mask):
  pixels = np.array(pixels, np.int32)
  test_mask = mask[pixels[..., 1], pixels[..., 0]]    # y, x
  # test_mask = grid_sample_depth_map(mask, pixels)    # y, x
  test_mask[no_match_mask] = 0
  return test_mask


def process_camera_pair(camera_pair, mask_pair, train_depth_map):
  train_camera, test_camera = camera_pair
  train_mask, gt_test_mask = mask_pair
  test_points, depth_list = sample_test_points(test_camera)
  best_depth_idx, best_pixels, no_match_mask = find_best_projected_points(test_points, train_camera, train_depth_map)
  best_depth = np.take_along_axis(depth_list, best_depth_idx[..., np.newaxis], axis=-1)
  best_depth = np.squeeze(best_depth, axis=-1)

  test_mask = get_mask_from_pixels(train_mask, best_pixels, no_match_mask)
  test_mask = cv2.morphologyEx(test_mask, cv2.MORPH_OPEN, np.ones([5, 5]))

  # visualize
  train_depth_map = train_depth_map / np.max(train_depth_map)
  best_depth = best_depth / np.max(train_depth_map)

  # train_depth_map = train_depth_map * 2
  # best_depth = best_depth * 2

  train_image = np.concatenate([train_mask.squeeze(), train_depth_map], axis=0)
  test_image = np.concatenate([test_mask.squeeze(), best_depth], axis=0)
  gt_test_image = np.concatenate([gt_test_mask.squeeze(), gt_test_mask.squeeze() * 0], axis=0)
  image = np.concatenate([train_image, test_image, gt_test_image], axis=1)
  cv2.imshow('image', image)
  cv2.waitKey()
  cv2.destroyAllWindows()


def main():
  datasource, config_dict = load_checkpoint()
  camera_pairs, mask_pairs = load_camera_pairs_with_masks(datasource, config_dict)
  depth_map_list = load_depth_map_list()
  frame_step = cal_frame_step(depth_map_list, camera_pairs)

  for i in tqdm(range(len(depth_map_list))):
    train_depth_map = depth_map_list[i]
    camera_idx = frame_step * i
    camera_pair = camera_pairs[camera_idx]
    mask_pair = mask_pairs[camera_idx]
    process_camera_pair(camera_pair, mask_pair, train_depth_map)


if __name__ == "__main__":
  main()