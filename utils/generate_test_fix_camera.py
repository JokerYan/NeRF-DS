import os
import shutil
from glob import glob

camera_idx = 93

if os.path.exists('/hdd/zhiwen/data/hypernerf/raw/'):
    data_root = '/hdd/zhiwen/data/hypernerf/raw/'
elif os.path.exists('/home/zwyan/3d_cv/data/hypernerf/raw/'):
    data_root = '/home/zwyan/3d_cv/data/hypernerf/raw/'
else:
    raise NotImplemented
# dataset = 'americano'
# dataset = 'white-board-6'
# dataset = 'aluminium-sheet-6'
# dataset = 'vrig-chicken'
dataset = 'vrig-cup-3_qualitative'
data_dir = os.path.join(data_root, dataset)

train_camera_folder = os.path.join(data_dir, "camera")
test_camera_folder = os.path.join(data_dir, "fix_camera_{}".format(camera_idx))

train_camera_name_list = []
for file_path in glob(os.path.join(train_camera_folder, "*")):
  filename = file_path.split('/')[-1]
  train_camera_name_list.append(filename)
train_camera_name_list = sorted(train_camera_name_list)

if 'left' in train_camera_name_list[0] or 'right' in train_camera_name_list[0]:
  left_camera_name_list = []
  right_camera_name_list = []
  for filename in train_camera_name_list:
    if 'left' in filename:
      left_camera_name_list.append(filename)
    elif 'right' in filename:
      right_camera_name_list.append(filename)
    else:
      raise Exception
  train_camera_name_list = left_camera_name_list

reference_camera_path = os.path.join(train_camera_folder, train_camera_name_list[camera_idx])
os.makedirs(test_camera_folder, exist_ok=True)
for filename in train_camera_name_list:
  target_camera_path = os.path.join(test_camera_folder, filename)
  shutil.copy(reference_camera_path, target_camera_path)
