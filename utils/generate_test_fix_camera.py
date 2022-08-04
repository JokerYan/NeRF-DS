import os
import shutil
from glob import glob

camera_idx = 322

if os.path.exists('/hdd/zhiwen/data/hypernerf/raw/'):
    data_root = '/hdd/zhiwen/data/hypernerf/raw/'
elif os.path.exists('/home/zwyan/3d_cv/data/hypernerf/raw/'):
    data_root = '/home/zwyan/3d_cv/data/hypernerf/raw/'
else:
    raise NotImplemented
data_dir = os.path.join(data_root, 'americano/')

train_camera_folder = os.path.join(data_dir, "camera")
test_camera_folder = os.path.join(data_dir, "fix_camera_{}".format(camera_idx))

train_camera_name_list = []
for file_path in glob(train_camera_folder + "/*"):
  filename = file_path.split('/')[-1]
  train_camera_name_list.append(filename)
train_camera_name_list = sorted(train_camera_name_list)

reference_camera_path = os.path.join(train_camera_folder, train_camera_name_list[camera_idx])
os.makedirs(test_camera_folder, exist_ok=True)
for filename in train_camera_name_list:
  target_camera_path = os.path.join(test_camera_folder, filename)
  shutil.copy(reference_camera_path, target_camera_path)