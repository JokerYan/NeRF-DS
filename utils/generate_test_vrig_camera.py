import os
import shutil
import json
from glob import glob

if os.path.exists('/hdd/zhiwen/data/hypernerf/raw/'):
    data_root = '/hdd/zhiwen/data/hypernerf/raw/'
elif os.path.exists('/home/zwyan/3d_cv/data/hypernerf/raw/'):
    data_root = '/home/zwyan/3d_cv/data/hypernerf/raw/'
else:
    raise NotImplemented
# dataset = 'americano'
dataset = '000_bell_01_novel_view'
data_dir = os.path.join(data_root, dataset)

train_camera_folder = os.path.join(data_dir, "camera")
test_camera_folder = os.path.join(data_dir, "vrig_camera")

dataset_info_path = os.path.join(data_dir, 'dataset.json')

with open(dataset_info_path, 'r') as f:
  dataset_info = json.load(f)
  val_camera_name_list = dataset_info['val_ids']

os.makedirs(test_camera_folder, exist_ok=True)
for val_id in val_camera_name_list:
  try:
    val_id_number = val_id.split('_')[1]
    int(val_id_number)
  except:
    val_id_number = val_id.split('_')[0]
    int(val_id_number)
  reference_camera_path = os.path.join(train_camera_folder, val_id + ".json")
  target_camera_path = os.path.join(test_camera_folder, val_id_number + ".json")
  shutil.copy(reference_camera_path, target_camera_path)


# train_camera_name_list = []
# for file_path in glob(os.path.join(train_camera_folder, "*")):
#   filename = file_path.split('/')[-1]
#   train_camera_name_list.append(filename)
# train_camera_name_list = sorted(train_camera_name_list)
#
# assert train_camera_name_list[0].startswith('left') or train_camera_name_list[0].startswith('right')
# left_camera_name_list = []
# right_camera_name_list = []
# for filename in train_camera_name_list:
#   if filename.startswith('left'):
#     left_camera_name_list.append(filename)
#   elif filename.startswith('right'):
#     right_camera_name_list.append(filename)
#   else:
#     raise Exception
#
# for i in range(len(left_camera_name_list)):
#   if i % 2 == 0:
#     train_camera_name_list.append(right_camera_name_list[i])
#   else:
#     train_camera_name_list.append(left_camera_name_list[i])
#
# os.makedirs(test_camera_folder, exist_ok=True)
# for filename in train_camera_name_list:
#   filename_idx = filename.split('_')[-1]
#   reference_camera_path = os.path.join(train_camera_folder, filename)
#   target_camera_path = os.path.join(test_camera_folder, filename_idx)
#   shutil.copy(reference_camera_path, target_camera_path)
