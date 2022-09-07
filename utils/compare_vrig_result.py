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
# dataset = 'single-vrig-chicken'
# dataset = 'vrig-white-board-1_novel_view'
dataset = 'vrig-cup-2_novel_view'
data_dir = os.path.join(data_root, dataset)

video_render_step = 9
target_height = 720

# experiment_name_list = ['chicken_spec_exp01_base', 'chicken_spec_exp03']
# experiment_name_list = ['s_chicken_spec_exp01_base']
# experiment_name_list = ['vwb1_nv_hc_exp02']
experiment_name_list = ['vc2_nv_hc_exp01', 'vc2_nv_ref_exp01']

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
for i in range(0, len(val_ids), video_render_step):
  eval_id = val_ids[i]
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
    target_width = frame.shape[1] * target_height // frame.shape[0]
    frame = cv2.resize(frame, (target_width, target_height))
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
frame_idx = 0
while True:
  full_image = None
  for exp_id in range(len(exp_concat_image_list)):
    exp_image = exp_concat_image_list[exp_id][frame_idx]
    if full_image is None:
      full_image = exp_image
    else:
      full_image = np.concatenate([full_image, exp_image], axis=0)
  actual_idx, total_idx = frame_idx * video_render_step + 1, len(exp_concat_image_list[0]) * video_render_step + 1
  full_image = cv2.putText(full_image, "{}/{}".format(actual_idx, total_idx),
                           (10, target_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
  cv2.imshow('full', full_image)
  key = cv2.waitKey()
  if key == ord('q'):
    break
  elif key == ord('1'):
    frame_idx = max(0, frame_idx - 1)
  elif key == ord('3'):
    frame_idx = min(len(exp_concat_image_list[0]) - 1, frame_idx + 1)
  elif key == ord('4'):
    frame_idx = max(0, frame_idx - 10)
  elif key == ord('6'):
    frame_idx = min(len(exp_concat_image_list[0]) - 1, frame_idx + 10)

cv2.destroyAllWindows()
