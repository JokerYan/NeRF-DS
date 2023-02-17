import os

import cv2
import numpy as np
from load_results import load_gt, load_output


interval = 1
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
  "americano_masked"
]
exp_configs = [
  # ("ms", "exp40"),
  # ("ref", "exp01"),
  # ("mso", "exp01"),
  # ("base", "exp01"),
  # ("nerfies", "exp01"),
  # ("refnerf", ""),
  # ("ms", "exp41"),

  # ("ref", "exp03"),
  ("base", "exp01"),
  ("ms", "exp40")
]

output_types = [
  # 'med_depth', 'med_points', 'ray_delta_x', 'ray_norm', 'ray_predicted_mask', 'rgb'
  # 'rgb', 'ray_norm', 'ray_predicted_mask'
  # 'ray_delta_x'
  # 'rgb', "ray_rotation_field"
  # 'rgb', "ray_norm", "ray_rotation_field"
  'rgb', 'med_depth'
]
out_dir = '/home/zwyan/3d_cv/repos/hypernerf_barf/evaluations/images'

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
  image = cv2.putText(image, text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
  return image


def select_visualization_for_dataset(dataset, selected_exp_config):
  dataset_nv = f'{dataset}_novel_view'
  images_list = []
  name_list = []

  gt_images = load_gt(dataset_nv)
  if interval > 0:
    gt_images_selected = []
    for i in range(0, len(gt_images), interval):
      gt_images_selected.append(gt_images[i])
    gt_images = gt_images_selected
  images_list.append(gt_images)
  name_list.append('gt')

  config_key, exp_idx = selected_exp_config
  for output_type in output_types:
    out_images = load_output(dataset, config_key, exp_idx, output_type, skip=interval > 1, vrig=dataset.startswith('0'))
    name_list.append(f"{config_key}_{exp_idx}_{output_type}")
    images_list.append(out_images)

  # display
  idx = 0
  while True:
    cur_image_list = [images[idx] for images in images_list]
    cur_image_list_with_text = [
      add_text_to_image(images[idx], name_list[i] + " " + str(idx)) for i, images in enumerate(images_list)
    ]
    full_image_with_text = concat_images(cur_image_list_with_text, gap=10)
    cv2.imshow('', full_image_with_text)
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
      # full_image = concat_images(cur_image_list, gap=10)
      # image_name = f'{dataset}_{idx}.png'
      # image_path = os.path.join(out_dir, image_name)
      # cv2.imwrite(image_path, full_image)
      for i, cur_image in enumerate(cur_image_list):
        name = name_list[i]
        image_name = f'{dataset}_{idx}_{name}.png'
        image_path = os.path.join(out_dir, image_name)
        cv2.imwrite(image_path, cur_image)
  cv2.destroyAllWindows()


if __name__ == "__main__":
  dataset_idx = 4
  # exp_idx = 3
  exp_idx = 1
  selected_exp_config = exp_configs[exp_idx]
  select_visualization_for_dataset(dataset_list[dataset_idx], selected_exp_config)
  print(f"Visualizations saved to: {out_dir}")