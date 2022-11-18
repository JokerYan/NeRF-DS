import os

import cv2
import numpy as np
from load_results import load_gt, load_output


dataset_list = [
  "011_bell_07",
  "015_cup_02",
  "018_as_01",
  "021_basin_01",
  "022_sieve_02",
  "025_press_01",
  "026_bowl_02",
  "028_plate_03",
  "029_2cup_01",
]
exp_configs = [
  ("ms", "exp40"),
  ("ref", "exp01"),
  ("mso", "exp01"),
  ("base", "exp01"),
  ("nerfies", "exp01"),
  ("refnerf", ""),
]
exp_name_list = [
  "NeRF-DS (Ours)",
  "NeRF-DS w/o Mask",
  "NeRF-DS w/o Surface",
  "HyperNeRF",
  "Nerfies",
  "Ref-NeRF"
]
out_dir_default = '/home/zwyan/3d_cv/repos/hypernerf_barf/evaluations/videos'

def concat_images(images, gap=0):
  height, width, channel = images[0].shape
  if gap > 0:
    gap_image = np.ones([height, gap, channel], dtype=np.uint8) * 255
    images_with_gap = []
    for i in range(len(images)):
      images_with_gap.append(images[i])
      if i < len(images) - 1:
        images_with_gap.append(gap_image)
    images = images_with_gap
  concat_image = np.concatenate(images, axis=1)
  return concat_image


def add_text_to_image(image, text):
  height, width, channel = image.shape
  image = np.copy(image)
  image = cv2.putText(image, text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
  return image


def select_rgb_for_dataset(dataset, selected_exp_configs, selected_exp_names, name_prefix=None):
  dataset_nv = f'{dataset}_novel_view'
  images_list = []
  name_list = []
  selected_exp_names = ['Novel View (GT)'] + selected_exp_names

  gt_images = load_gt(dataset_nv)
  images_list.append(gt_images)
  name_list.append('gt')

  for i, (config_key, exp_idx) in enumerate(selected_exp_configs):
    out_images = load_output(dataset, config_key, exp_idx)
    images_list.append(out_images)
    name_list.append(f"{config_key}_{exp_idx}")

  for i in range(len(images_list)):
    images = images_list[i]
    for j in range(len(images)):
      image_with_text = add_text_to_image(images[j], selected_exp_names[i])
      images[j] = image_with_text

  full_image_list = []
  for i in range(len(images_list[0])):
    exp_images = [images[i] for images in images_list]
    full_image = concat_images(exp_images, gap=10)
    full_image_list.append(full_image)

  # display
  idx = 0
  while True:
    cv2.imshow('', full_image_list[idx])
    key = cv2.waitKey()
    if key == ord('q'):
      break
    elif key == ord('1'):
      idx = max(0, idx - 1)
    elif key == ord('3'):
      idx = min(len(images_list[0]) - 1, idx + 1)
    elif key == ord('4'):
      idx = max(0, idx - 10)
    elif key == ord('6'):
      idx = min(len(images_list[0]) - 1, idx + 10)
    elif key == ord('s'):
      video_name = f'{dataset}_qualitative.mp4'
      os.makedirs(out_dir_default, exist_ok=True)
      video_path = os.path.join(out_dir_default, video_name)
      frames = full_image_list
      _fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
      height, width = frames[0].shape[:2]
      video_writer = cv2.VideoWriter(video_path, _fourcc, 30.0, (width, height))
      for frame in frames:
        video_writer.write(frame)
      video_writer.release()
      print("Video saved to {}".format(video_path))
      break
  cv2.destroyAllWindows()


if __name__ == "__main__":
  dataset_idx = 7
  exp_idx_list = [0, 3, 4, 5]   # vs baseline
  name_prefix = None

  # exp_idx_list = [0, 1]   # vs ablation
  # name_prefix = 'ablation_ref'

  # exp_idx_list = [0, 2]   # vs ablation
  # name_prefix = 'ablation_mso'

  selected_exp_configs = [exp_configs[i] for i in exp_idx_list]
  selected_exp_names = [exp_name_list[i] for i in exp_idx_list]
  select_rgb_for_dataset(dataset_list[dataset_idx], selected_exp_configs, selected_exp_names, name_prefix)
