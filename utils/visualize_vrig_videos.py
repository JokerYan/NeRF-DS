import os
import cv2
import json
import numpy as np
from glob import glob

if os.path.exists('/hdd/zhiwen/data/hypernerf/raw/'):
    data_root = '/hdd/zhiwen/data/hypernerf/raw/'
    experiment_root = '/hdd/zhiwen/hypernerf_barf/experiments/'
elif os.path.exists('/home/zwyan/3d_cv/data/hypernerf/raw/'):
    data_root = '/home/zwyan/3d_cv/data/hypernerf/raw/'
    experiment_root = '/home/zwyan/3d_cv/repos/hypernerf_barf/experiments/'
else:
    raise NotImplemented

dataset = 'vrig-bell-1_novel_view'
data_dir = os.path.join(data_root, dataset)

# load gt
image_dir = os.path.join(data_dir, "rgb/1x/")
print(image_dir)
filenames = []
for filename in glob(os.path.join(image_dir, "*.png")):
    filenames.append(filename)
filenames.sort()

for i in range(len(filenames) // 2):
    left_image_path = filenames[i * 2]
    right_image_path = filenames[i * 2 + 1]
    left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)
    full_image = np.concatenate([left_image, right_image], axis=1)

    cv2.imshow("full image", full_image)
    key = cv2.waitKey()
    if key == ord("q"):
        break