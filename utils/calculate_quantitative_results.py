import os
import cv2
import json
import numpy as np
import lpips
import torch
import tensorflow as tf
from tqdm import tqdm

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
dataset = '011_bell_07_novel_view'
data_dir = os.path.join(data_root, dataset)
experiment_name = "011_b07_nv_base_exp01"

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

print("mse: {:.5f} psnr: {:.3f} ms_ssim: {:.3f} lpips: {:.3f}".format(
  np.mean(mse_list), np.mean(psnr_list), np.mean(ms_ssim_list), np.mean(lpips_list)))

# base
# mse: 0.00618 psnr: 22.507 ms_ssim: 0.936 lpips: 0.120
# mse: 0.00314 psnr: 25.175 ms_ssim: 0.952 lpips: 0.099   # trim 0.1
# mse: 0.00613 psnr: 22.527 ms_ssim: 0.938 lpips: 0.122   # full

# ref
# mse: 0.00615 psnr: 22.513 ms_ssim: 0.934 lpips: 0.126
# mse: 0.00328 psnr: 25.002 ms_ssim: 0.950 lpips: 0.105   # trim 0.1

# hsf
# mse: 0.00559 psnr: 22.887 ms_ssim: 0.936 lpips: 0.112
# mse: 0.00290 psnr: 25.514 ms_ssim: 0.953 lpips: 0.091   # trim 0.1
# mse: 0.00553 psnr: 22.913 ms_ssim: 0.938 lpips: 0.115   # full
