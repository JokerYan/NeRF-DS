import os
import shutil
from glob import glob

data_dir = "/hdd/zhiwen/data/hypernerf/raw/americano"
train_camera_folder = os.path.join(data_dir, "camera")
test_camera_folder = os.path.join(data_dir, "fix_camera")

train_camera_name_list = []
for file_path in glob(train_camera_folder + "/*"):
  filename = file_path.split('/')[-1]
  train_camera_name_list.append(filename)
train_camera_name_list = sorted(train_camera_name_list)

reference_camera_path = os.path.join(train_camera_folder, train_camera_name_list[0])
os.makedirs(test_camera_folder, exist_ok=True)
for filename in train_camera_name_list:
  target_camera_path = os.path.join(test_camera_folder, filename)
  shutil.copy(reference_camera_path, target_camera_path)