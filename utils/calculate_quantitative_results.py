import os
import cv2
import json
import numpy as np
import lpips
import torch
import tensorflow as tf
from tqdm import tqdm
from glob import glob

from hypernerf import image_utils
from hypernerf import utils


def trim_image(image: np.ndarray, ratio=0.1):
  height, width, _ = image.shape
  trim_h, trim_w = int(height * ratio), int(width * ratio)
  trimmed_image = image[trim_h:height-trim_h, trim_w:width-trim_w, :]

  return trimmed_image


def compute_multiscale_ssim(image1: np.ndarray, image2: np.ndarray):
  """Compute the multiscale SSIM metric."""
  image1 = tf.convert_to_tensor(image1)
  image2 = tf.convert_to_tensor(image2)
  return tf.image.ssim_multiscale(image1, image2, max_val=1.0)


loss_fn_alex = lpips.LPIPS(net='alex')
def compute_lpips(image1: np.ndarray, image2: np.ndarray):
  """Compute the LPIPS metric."""
  # normalize to -1 to 1
  image1 = image1 * 2 - 1
  image2 = image2 * 2 - 1

  # append batch
  image1 = image1.transpose([2, 0, 1])
  image2 = image2.transpose([2, 0, 1])
  image1 = image1[np.newaxis, ...]
  image2 = image2[np.newaxis, ...]

  image1 = torch.Tensor(image1)
  image2 = torch.Tensor(image2)

  lpips = loss_fn_alex(image1, image2)
  lpips = lpips.detach().numpy()
  return float(lpips)


if os.path.exists('/hdd/zhiwen/data/hypernerf/raw/'):
    data_root = '/hdd/zhiwen/data/hypernerf/raw/'
    experiment_root = '/hdd/zhiwen/hypernerf_barf/experiments/'
elif os.path.exists('/home/zwyan/3d_cv/data/hypernerf/raw/'):
    data_root = '/home/zwyan/3d_cv/data/hypernerf/raw/'
    experiment_root = '/home/zwyan/3d_cv/repos/hypernerf_barf/experiments/'
else:
    raise NotImplemented
refnerf_root = '/home/zwyan/3d_cv/repos/multinerf/experiments/spec/'


# dataset = '011_bell_07_novel_view'
# dataset = '012_cup_01_novel_view'
# dataset = '013_bowl_01_novel_view'
# dataset = '014_spoon_02_novel_view'
# dataset = '015_cup_02_novel_view'
# dataset = '016_spoon_03_novel_view'
# dataset = '017_cup_03_novel_view'
dataset = '018_as_01_novel_view'
# dataset = '019_plate_01_novel_view'
# dataset = '020_sieve_01_novel_view'
# dataset = '021_basin_01_novel_view'
# dataset = '022_sieve_02_novel_view'
# dataset = '025_press_01_novel_view'
# dataset = '026_bowl_02_novel_view'
# dataset = '027_dryer_01_novel_view'
# dataset = '028_plate_03_novel_view'
# dataset = '029_2cup_01_novel_view'

data_dir = os.path.join(data_root, dataset)

experiment_name = 'refnerf'
# experiment_name = "011_b07_nv_ms_exp40"
# experiment_name = "013_bo01_nv_base_exp01"
# experiment_name = "014_s02_nv_ms_exp36"
# experiment_name = "015_c02_nv_nerfies_exp01"
# experiment_name = "016_s03_nv_ms_exp40"
# experiment_name = "017_c03_nv_ms_exp39"
# experiment_name = "018_a01_nv_ms_exp40"
# experiment_name = "019_p01_nv_ms_exp39"
# experiment_name = "020_sv01_nv_base_exp01"
# experiment_name = "021_bs01_nv_nerfies_exp01"
# experiment_name = "022_sv02_nv_base_exp01"
# experiment_name = "025_ps01_nv_base_exp01"
# experiment_name = "026_bo02_nv_base_exp01"
# experiment_name = "027_dr01_nv_ref_exp01"
# experiment_name = "028_p03_nv_base_exp01"
# experiment_name = "029_2c01_nv_base_exp01"

skip = True
if skip:
  video_render_step = 9
else:
  video_render_step = 1
image_scale = 1
trim_ratio = 0

# load gt
dataset_info_dir = os.path.join(data_dir, 'dataset.json')
rgb_dir = os.path.join(data_dir, 'rgb', '{}x'.format(image_scale))

with open(dataset_info_dir, 'r') as f:
  dataset_info = json.load(f)
val_ids = dataset_info['val_ids']
gt_images = []
for i in range(0, len(val_ids), video_render_step):
  eval_id = val_ids[i]
  gt_path = os.path.join(rgb_dir, eval_id + ".png")
  gt_image = cv2.imread(gt_path)
  gt_images.append(gt_image)
target_shape = gt_images[0].shape

# load rendered video
frame_list = []
if experiment_name == 'refnerf':
  assert dataset.endswith('_novel_view')
  dataset = dataset[:-11]
  experiment_name = f'{dataset}_refnerf'
  experiment_dir = os.path.join(refnerf_root, experiment_name)
  render_dir = os.path.join(refnerf_root, experiment_name, 'render', 'test_preds_step_250000', 'color_*.png')
  filepath_list = []
  for filepath in glob(render_dir):
    filepath_list.append(filepath)
  filepath_list = sorted(filepath_list)

  for filepath in filepath_list:
    frame = cv2.imread(filepath)
    frame_list.append(frame)

else:
  experiment_dir = os.path.join(experiment_root, experiment_name)
  if skip:
    video_name = 'result_vrig_camera.mp4'
  else:
    video_name = 'result_vrig_camera_full.mp4'
  video_path = os.path.join(experiment_dir, video_name)

  cap = cv2.VideoCapture(video_path)
  frame_list = []
  while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
      break
    if frame.shape == target_shape:
      pass
    elif frame.shape == tuple(target_shape * np.array([2, 4, 1])):
      frame = frame[:target_shape[0], :target_shape[1], :]
    elif frame.shape == tuple(target_shape * np.array([2, 3, 1])):
      frame = frame[:target_shape[0], :target_shape[1], :]
    else:
      raise Exception
    frame_list.append(frame)

assert len(gt_images) == len(frame_list), (len(gt_images), len(frame_list))
assert gt_images[0].shape == frame_list[0].shape, (gt_images[0].shape, frame_list[0].shape)

mse_list = []
psnr_list = []
ms_ssim_list = []
lpips_list = []
for i in tqdm(range(len(gt_images))):
  gt_image = gt_images[i]
  rendered_image = frame_list[i]

  # normalize to 1
  gt_image = gt_image / 255.0
  rendered_image = rendered_image / 255.0

  if trim_ratio > 0:
    gt_image = trim_image(gt_image, trim_ratio)
    rendered_image = trim_image(rendered_image, trim_ratio)

  mse = ((rendered_image - gt_image)**2).mean()
  psnr = utils.compute_psnr(mse)
  ms_ssim = compute_multiscale_ssim(gt_image, rendered_image)
  lpips = compute_lpips(gt_image, rendered_image)

  mse_list.append(mse)
  psnr_list.append(psnr)
  ms_ssim_list.append(ms_ssim)
  lpips_list.append(lpips)

result_str = "mse: {:.5f} psnr: {:.3f} ms_ssim: {:.3f} lpips: {:.3f}".format(
  np.mean(mse_list), np.mean(psnr_list), np.mean(ms_ssim_list), np.mean(lpips_list))
print(result_str)

# save results to txt
save_path = os.path.join(experiment_dir, 'quantitative_results.txt')
with open(save_path, 'w+') as f:
  f.write(result_str)
