import os
import cv2
import numpy as np

data_folder = '/home/zwyan/3d_cv/repos/hypernerf_barf/experiments/'
# output_path = os.path.join(data_folder, '011_b07_nv_ms_exp36', 'render_result_vrig_camera_full')
# output_path = os.path.join(data_folder, '015_c02_nv_ms_exp36', 'render_result_vrig_camera_full')
# output_path = os.path.join(data_folder, '015_c02_nv_ref_exp01', 'render_result_vrig_camera')

# output_path = os.path.join(data_folder, '015_c02_nv_ms_exp40', 'render_result_fix_camera_93')
# output_path = os.path.join(data_folder, '015_c02_nv_ref_exp01', 'render_result_fix_camera_93')

output_path = os.path.join(data_folder, '021_bs01_nv_ms_exp36', 'render_result_vrig_camera_full')
# output_path = os.path.join(data_folder, '021_bs01_nv_ref_exp01', 'render_result_vrig_camera_full')

# output_path = os.path.join(data_folder, '028_p03_nv_ms_exp40', 'render_result_vrig_camera')

interval = 9 if output_path.endswith('full') else 1

render_output = np.load(output_path, allow_pickle=True)
med_points = np.array([x['med_points'] for x in render_output])

delta_x = np.array([x['ray_delta_x'] for x in render_output])

canonical_points = med_points[..., :3].squeeze()

img = []
raw_img = []
aux_img = []
def click_event(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    coord = [(x, y)]
    print(raw_img[y, x, :])
    # print(aux_img[y, x, :])
    return coord


# lower_bound = np.array([-0.2, -0.2, -0.3])
# upper_bound = np.array([0.2, 0, 0.3])
# lower_bound = np.array([-0.3, -0.1, -0.3])
# upper_bound = np.array([-0.1, 0.1, -0.1])
lower_bound = np.array([0, -0.2, -0.3])
upper_bound = np.array([0.4, 0, 0])

i = 0
while i < len(canonical_points):
  img = canonical_points[i, ...]

  # img = img * 3 + np.array([0, 0, 1])
  raw_img = img
  img = (img - lower_bound) / (upper_bound - lower_bound)
  aux_img = delta_x[i, ...]

  img = cv2.putText(img, "{}/{}".format(i // interval, len(canonical_points) // interval),
                           (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
  cv2.imshow('canonical', img)
  cv2.setMouseCallback("canonical", click_event)
  key = cv2.waitKey()
  if key == ord('q'):
    cv2.destroyAllWindows()
    exit()
  elif key == ord('3'):
    i += interval
  elif key == ord('1'):
    i -= interval
  i = min(max(i, 0), len(canonical_points) - 1)
