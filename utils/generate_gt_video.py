import os
import mediapy
from glob import glob

fps = 30

if os.path.exists('/hdd/zhiwen/data/hypernerf/raw/'):
    data_root = '/hdd/zhiwen/data/hypernerf/raw/'
elif os.path.exists('/home/zwyan/3d_cv/data/hypernerf/raw/'):
    data_root = '/home/zwyan/3d_cv/data/hypernerf/raw/'
else:
    raise NotImplemented

# dataset = 'vrig-chicken'
dataset = 'aleks-teapot'
data_dir = os.path.join(data_root, dataset)

rgb_folder = os.path.join(data_dir, "rgb/1x")
if not os.path.isdir(rgb_folder):
  rgb_folder = os.path.join(data_dir, "rgb/2x")

if dataset.startswith('vrig'):
  filename_regex = "left*"
else:
  filename_regex = "*"

image_paths = []
for file_path in glob(os.path.join(rgb_folder, filename_regex)):
  image_paths.append(file_path)
image_paths = sorted(image_paths)

images = []
for file_path in image_paths:
  image = mediapy.read_image(file_path)
  images.append(image)

mediapy.set_show_save_dir(data_dir)
mediapy.show_video(images,fps=fps, title="gt")