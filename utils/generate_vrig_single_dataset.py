import os
import shutil
import json
import numpy as np

if os.path.exists('/hdd/zhiwen/data/hypernerf/raw/'):
    data_root = '/hdd/zhiwen/data/hypernerf/raw/'
    experiment_root = '/hdd/zhiwen/hypernerf_barf/experiments/'
elif os.path.exists('/home/zwyan/3d_cv/data/hypernerf/raw/'):
    data_root = '/home/zwyan/3d_cv/data/hypernerf/raw/'
    experiment_root = '/home/zwyan/3d_cv/repos/hypernerf_barf/experiments/'
else:
    raise NotImplemented

dataset = 'vrig-chicken'
data_dir = os.path.join(data_root, dataset)

target_dataset = 'single-' + dataset
target_data_dir = os.path.join(data_root, target_dataset)

dataset_info_path = os.path.join(data_dir, 'dataset.json')
target_dataset_info_path = os.path.join(target_data_dir, 'dataset.json')

if not os.path.isdir(target_data_dir):
    shutil.copytree(data_dir, target_data_dir)

# change dataset.json
with open(dataset_info_path, 'r') as f:
    dataset_json = json.load(f)
ids = dataset_json['ids']
left_ids = []
right_ids = []
for id in ids:
    if id.startswith('left'):
        left_ids.append(id)
    elif id.startswith('right'):
        right_ids.append(id)
    else:
        raise NotImplemented
assert 'train_ids' in dataset_json and 'val_ids' in dataset_json
dataset_json['train_ids'] = left_ids
dataset_json['val_ids'] = right_ids
with open(target_dataset_info_path, 'w') as f:
    json.dump(dataset_json, f, indent=4)