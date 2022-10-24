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
dataset = '019_plate_01_novel_view'
data_dir = os.path.join(data_root, dataset)

all_camera_folder = os.path.join(data_dir, "camera")
train_camera_folder = os.path.join(data_dir, "train_camera")
test_camera_folder = os.path.join(data_dir, "vrig_camera")

dataset_info_path = os.path.join(data_dir, 'dataset.json')

with open(dataset_info_path, 'r') as f:
  dataset_info = json.load(f)
  train_camera_name_list = dataset_info['train_ids']
  val_camera_name_list = dataset_info['val_ids']

# copy training cameras
os.makedirs(train_camera_folder, exist_ok=True)
for train_id in train_camera_name_list:
  try:
    train_id_number = train_id.split('_')[1]
    int(train_id_number)
  except:
    train_id_number = train_id.split('_')[0]
    int(train_id_number)
  reference_camera_path = os.path.join(all_camera_folder, train_id + ".json")
  target_camera_path = os.path.join(train_camera_folder, train_id + ".json")
  shutil.copy(reference_camera_path, target_camera_path)

# copy test cameras
os.makedirs(test_camera_folder, exist_ok=True)
for val_id in val_camera_name_list:
  try:
    val_id_number = val_id.split('_')[1]
    int(val_id_number)
  except:
    val_id_number = val_id.split('_')[0]
    int(val_id_number)
  reference_camera_path = os.path.join(all_camera_folder, val_id + ".json")
  target_camera_path = os.path.join(test_camera_folder, val_id + ".json")
  shutil.copy(reference_camera_path, target_camera_path)

