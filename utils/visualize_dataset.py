import os
import cv2
import numpy as np

from load_results import load_gt

dataset_list = [
  "021_basin_01",
  "028_plate_03",
  "018_as_01",
  "022_sieve_02",
  "011_bell_07",
  "015_cup_02",
  "025_press_01",
  "029_2cup_01",
]

out_dir = '/home/zwyan/3d_cv/repos/hypernerf_barf/evaluations/images'


def add_text_to_image(image, text):
  height, width, channel = image.shape
  image = np.copy(image)
  image = cv2.putText(image, text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
  return image


def concat_images(images, horizontal=True, gap=0):
  height, width, channel = images[0].shape
  if gap > 0:
    if horizontal:
      gap_image = np.ones([height, gap, channel], dtype=np.uint8) * 255
    else:
      gap_image = np.ones([gap, width, channel], dtype=np.uint8) * 255
    images_with_gap = []
    for i in range(len(images)):
      images_with_gap.append(images[i])
      if i < len(images) - 1:
        images_with_gap.append(gap_image)
    images = images_with_gap
  concat_axis = 1 if horizontal else 0
  concat_image = np.concatenate(images, axis=concat_axis)
  return concat_image


def load_dataset_images(dataset_name, num_images, horizontal_gap=0):
  train_images = load_gt(f'{dataset_name}_novel_view', train=True)
  test_images = load_gt(f'{dataset_name}_novel_view', train=False)
  skip = len(train_images) // num_images
  skip_dataset_images = []
  for i in range(len(train_images)):
    if i % skip == 0 and len(skip_dataset_images) < num_images:
      skip_image_train = train_images[i]
      skip_image_test = test_images[i]
      skip_image = concat_images([skip_image_train, skip_image_test], horizontal=False)
      skip_image = add_text_to_image(skip_image, f't={i}')
      skip_dataset_images.append(skip_image)

  # concat horizontally
  dataset_image = concat_images(skip_dataset_images, horizontal=True, gap=horizontal_gap)
  return dataset_image


def visualize_datasets(dataset_name_list, num_images, horizontal_gap=0, vertical_gap=0):
  dataset_image_list = []
  for dataset_name in dataset_name_list:
    dataset_image = load_dataset_images(dataset_name, num_images, horizontal_gap=horizontal_gap)
    dataset_image_list.append(dataset_image)
  full_image = concat_images(dataset_image_list, horizontal=False, gap=vertical_gap)

  cv2.imshow('', full_image)
  key = cv2.waitKey()
  if key == ord('s'):
    out_path = os.path.join(out_dir, 'dataset_visualization.png')
    cv2.imwrite(out_path, full_image)
  cv2.destroyAllWindows()


if __name__ == "__main__":
  visualize_datasets(dataset_list, 8, horizontal_gap=10, vertical_gap=20)