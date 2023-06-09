import json
import os
import cv2
import numpy as np
from glob import glob

from hypernerf import model_utils

raw_data_root = '/home/zwyan/3d_cv/data/hypernerf/raw'
exp_root = '/home/zwyan/3d_cv/repos/hypernerf_barf/experiments'
refnerf_root = '/home/zwyan/3d_cv/repos/multinerf/experiments'


def image_np_to_cv2(image, norm_vector=False, norm_to_one=False, absolute=False, scale=1):
  if norm_to_one:
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
  if scale != 1:
    image = image * scale
  if norm_vector:
    image = np.array(model_utils.normalize_vector(image))
    image = (image / 2.0 + 0.5)
  if absolute:
    image = np.abs(image)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = np.round(image * 255)
  image = np.clip(image, 0, 255)
  image = np.array(image, dtype=np.uint8)
  return image


def load_gt(dataset_name, scale=1, train=False):
  data_dir = os.path.join(raw_data_root, dataset_name)
  rgb_dir = os.path.join(data_dir, 'rgb', f'{scale}x')
  camera_name = 'left' if train else 'right'
  filename_format = f'*_{camera_name}.png'
  rgb_glob = os.path.join(rgb_dir, filename_format)
  image_list = []
  for image_path in sorted(glob(rgb_glob)):
    image = cv2.imread(image_path)
    image_list.append(image)

    # cv2.imshow('', image)
    # key = cv2.waitKey()
    # if key == ord('q'):
    #   cv2.destroyAllWindows()
    #   break
  return image_list


def load_hypernerf_gt(daset_name, scale=2, train=False):
  data_dir = os.path.join(raw_data_root, daset_name)
  rgb_dir = os.path.join(data_dir, 'rgb', f'{scale}x')
  info_json_path = os.path.join(data_dir, 'dataset.json')
  with open(info_json_path, 'r') as f:
    info_json = json.load(f)
  if train:
    ids = info_json['train_ids']
  else:
    ids = info_json['val_ids']

  image_list = []
  for id in ids:
    image_path = os.path.join(rgb_dir, f'{id}.png')
    image = cv2.imread(image_path)
    image_list.append(image)
  return image_list

def load_hypernerf(exp_prefix, config_key, exp_idx, output_type='rgb', skip=False, vrig=True):
  exp_name = f'{exp_prefix}_{config_key}_{exp_idx}'
  exp_dir = os.path.join(exp_root, exp_name)
  if vrig:
    result_name = 'render_result_vrig_camera_full' if not skip else 'render_result_vrig_camera'
  else:
    result_name = 'render_result_fix_camera_93' if not skip else 'render_result_fix_camera_93'
  exp_result_path = os.path.join(exp_dir, result_name)
  exp_result = np.load(exp_result_path, allow_pickle=True)

  # load output
  output_list = [x[output_type] for x in exp_result]
  if output_type == 'ray_rotation_field':
    for i in range(len(output_list)):
      print(np.min(output_list[i]), np.max(output_list[i]))
  norm_vector = output_type in ['ray_norm']
  # norm_to_one = output_type in ['ray_delta_x']
  norm_to_one = False
  absolute = output_type in ['ray_delta_x']
  # absolute = False
  # scale = 10000 if output_type in ['ray_delta_x'] else 1
  if output_type in ['ray_delta_x']:
    array_output = np.array(output_list)
    scale = 1 / (np.max(array_output) - np.min(array_output))
  else:
    scale = 1
  # print(output_list[0])
  output_list = [image_np_to_cv2(x, norm_vector=norm_vector, norm_to_one=norm_to_one,
                                 absolute=absolute, scale=scale) for x in output_list]

  # for image in output_list:
  #   cv2.imshow('', image)
  #   key = cv2.waitKey()
  #   if key == ord('q'):
  #     cv2.destroyAllWindows()
  #     break

  return output_list


def load_refnerf(dataset_name):
  exp_name = f'{dataset_name}_refnerf'
  exp_dir = os.path.join(refnerf_root, 'spec', exp_name)
  render_dir = os.path.join(exp_dir, 'render', 'test_preds_step_250000')
  render_glob = os.path.join(render_dir, 'color_*.png')

  image_list = []
  for image_path in sorted(glob(render_glob)):
    image = cv2.imread(image_path)
    image_list.append(image)

    # cv2.imshow('', image)
    # key = cv2.waitKey()
    # if key == ord('q'):
    #   cv2.destroyAllWindows()
    #   break

  return image_list


def load_output(dataset_name, config_key, exp_idx, output_type='rgb', skip=False, vrig=True):
    is_refnerf = config_key == 'refnerf'
    if is_refnerf:
      out_images = load_refnerf(dataset_name)
    else:
      if vrig:
        exp_prefix = dataset_name + '_nv'
      else:
        exp_prefix = dataset_name
      out_images = load_hypernerf(exp_prefix, config_key, exp_idx, output_type, skip, vrig)
    return out_images


if __name__ == "__main__":
  load_gt("011_bell_07_novel_view")
  # load_hypernerf("011_b07_nv", "ms", "exp40")
  # load_refnerf('011_bell_07')