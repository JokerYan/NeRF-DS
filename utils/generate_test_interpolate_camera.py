import os
import copy
import json
import shutil
import numpy as np
from glob import glob

data_dir = "/hdd/zhiwen/data/hypernerf/raw/americano"
train_camera_folder = os.path.join(data_dir, "camera")
test_camera_folder = os.path.join(data_dir, "interpolate_camera")

train_camera_name_list = []
for file_path in glob(train_camera_folder + "/*"):
  filename = file_path.split('/')[-1]
  train_camera_name_list.append(filename)
train_camera_name_list = sorted(train_camera_name_list)

reference_camera_path_first = os.path.join(train_camera_folder, train_camera_name_list[0])
reference_camera_path_second = os.path.join(train_camera_folder, train_camera_name_list[140])
reference_json_first = json.load(open(reference_camera_path_first, 'r'))
reference_json_second = json.load(open(reference_camera_path_second, 'r'))
position_first = reference_json_first['position']
position_second = reference_json_second['position']

os.makedirs(test_camera_folder, exist_ok=True)
for i, filename in enumerate(train_camera_name_list):
  target_camera_path = os.path.join(test_camera_folder, filename)
  target_json_template = copy.deepcopy(reference_json_first)
  target_position = (np.array(position_first) * (len(train_camera_name_list) - 1 - i)
                     + np.array(position_second) * i) / (len(train_camera_name_list) - 1)
  target_position = target_position.tolist()
  target_json_template['position'] = target_position
  json.dump(target_json_template, open(target_camera_path, 'w+'), indent=4)