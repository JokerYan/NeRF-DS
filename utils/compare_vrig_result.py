import os
import cv2
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
# dataset = 'americano'
# dataset = 'vrig-chicken'
dataset = 'single-vrig-chicken'
data_dir = os.path.join(data_root, dataset)

experiment_name_list = ['chicken_spec_exp01_base', 'chicken_spec_exp03']
# experiment_name_list = ['s_chicken_spec_exp01_base']
video_path_list = []
for experiment_name in experiment_name_list:
  experiment_dir = os.path.join(experiment_root, experiment_name)
  video_name = 'result_vrig_camera.mp4'
  video_path = os.path.join(experiment_dir, video_name)
  video_path_list.append(video_path)

dataset_info_dir = os.path.join(data_dir, 'dataset.json')
rgb_dir = os.path.join(data_dir, 'rgb', '1x')

# load gt
with open(dataset_info_dir, 'r') as f:
  dataset_info = json.load(f)
val_ids = dataset_info['val_ids']
gt_images = []
for eval_id in val_ids:
  gt_path = os.path.join(rgb_dir, eval_id + ".png")
  gt_image = cv2.imread(gt_path)
  gt_images.append(gt_image)

# load rendered video
exp_frame_list = []
for video_path in video_path_list:
  cap = cv2.VideoCapture(video_path)
  frame_list = []
  while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
      break
    frame_list.append(frame)
  exp_frame_list.append(frame_list)

# concatenate and show
exp_concat_image_list = []
for frame_list in exp_frame_list:
  assert len(frame_list) == len(gt_images)
  concat_image_list = []
  for i in range(len(gt_images)):
    gt_image = gt_images[i]
    render_image = frame_list[i]
    height = render_image.shape[0]
    width = int(gt_images[i].shape[1] * height / gt_images[i].shape[0])

    gt_image = cv2.resize(gt_image, (width, height))
    concat_image = np.concatenate([gt_image, render_image], axis=1)
    concat_image_list.append(concat_image)
  exp_concat_image_list.append(concat_image_list)


# concat image between experiments and show
for i in range(len(exp_concat_image_list[0])):
  full_image = None
  for exp_id in range(len(exp_concat_image_list)):
    exp_image = exp_concat_image_list[exp_id][i]
    if full_image is None:
      full_image = exp_image
    else:
      full_image = np.concatenate([full_image, exp_image], axis=0)
  cv2.imshow('full', full_image)
  if cv2.waitKey(0) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()
