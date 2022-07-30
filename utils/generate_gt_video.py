import os
import mediapy
from glob import glob

fps = 30
data_dir = "/hdd/zhiwen/data/hypernerf/raw/americano"
rgb_folder = os.path.join(data_dir, "rgb/1x")

images = []
for file_path in glob(rgb_folder + "/*"):
  image = mediapy.read_image(file_path)
  images.append(image)

mediapy.set_show_save_dir(data_dir)
mediapy.show_video(images,fps=fps, title="gt")