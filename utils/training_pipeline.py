import os
import re
import subprocess
import argparse
from tqdm import tqdm

# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=64'
"""
export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/white-board-6/
export EXPERIMENT_PATH=experiments/wb6_exp01
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_spec_ref.gin
"""

config_root = "./configs/"
config_dict = {
  'base': "test_local.gin",
  'vbase': "test_local_vrig.gin",
  "ref": "test_local_spec_ref.gin",
  "vref": "test_local_spec_ref_vrig.gin",
  "hc": "test_local_spec_hc.gin",
  "vhc": "test_local_spec_hc_vrig.gin",
  "hcx": "test_local_spec_hcx.gin",
  "hcx_nv": "test_local_spec_hcx_nv.gin",
  "vhcx_nv": "test_local_spec_hcx_nv_vrig.gin",
  "hcxt": "test_local_spec_hcxt.gin",
  "vhcxt": "test_local_spec_vhcxt.gin",

  "hs": "test_local_spec_hs.gin",
  "vhs": "test_local_spec_hs_vrig.gin",

  "hsf": "test_local_spec_hsf.gin",
  "ms": "test_local_spec_ms.gin",
}

data_root = ""
if os.path.isdir("/ssd/zhiwen/data/hypernerf/raw/"):
  data_root = "/ssd/zhiwen/data/hypernerf/raw/"
elif os.path.isdir("/hdd/zhiwen/data/hypernerf/raw/"):
  data_root = "/hdd/zhiwen/data/hypernerf/raw/"
elif os.path.isdir("/home/zwyan/3d_cv/data/hypernerf/raw/"):
  data_root = "/home/zwyan/3d_cv/data/hypernerf/raw/"

exp_root = "./experiments/"
# if os.path.isdir('/ssd/zhiwen/exp/hypernerf'):
#   exp_root = '/ssd/zhiwen/exp/hypernerf'

# training schedule in the tuple of dataset_name, exp_name, config_key, gin_bindings
training_schedule = [
  # ("011_bell_07_novel_view", "011_b07_nv_ms_exp12", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                          "NerfModel.use_predicted_mask = True",
  #                                                          "NerfModel.use_coarse_depth_for_mask = True"]),
  # ("011_bell_07_novel_view", "011_b07_nv_ms_exp13", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                          "NerfModel.use_predicted_mask = True",
  #                                                          "NerfModel.use_coarse_depth_for_mask = True"]),
  # ("012_cup_01_novel_view", "012_c01_nv_ms_exp13", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                          "NerfModel.use_predicted_mask = True",
  #                                                          "NerfModel.use_coarse_depth_for_mask = True"]),
  # ("011_bell_07_novel_view", "011_b07_nv_ms_exp14", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                          "NerfModel.use_predicted_mask = True",
  #                                                          "NerfModel.use_coarse_depth_for_mask = True"]),
  # ("011_bell_07_novel_view", "011_b07_nv_ms_exp15", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                          "NerfModel.use_predicted_mask = True",
  #                                                          "NerfModel.use_coarse_depth_for_mask = True",
  #                                                          "NerfModel.use_norm_voxel = True"]),

  # ("013_bowl_01_novel_view", "013_bo01_nv_ref_exp01", "ref", ["ExperimentConfig.image_scale = 1"]),
  # ("013_bowl_01_novel_view", "013_bo01_nv_ms_exp14", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                          "NerfModel.use_predicted_mask = True",
  #                                                          "NerfModel.use_coarse_depth_for_mask = True"]),


  # ("011_bell_07_novel_view", "011_b07_nv_ms_exp16", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                          "NerfModel.use_predicted_mask = True",
  #                                                          "NerfModel.use_coarse_depth_for_mask = True",
  #                                                          "NerfModel.use_x_in_rgb_condition = True"]),
  # ("012_cup_01_novel_view", "012_c01_nv_ms_exp14", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                          "NerfModel.use_predicted_mask = True",
  #                                                          "NerfModel.use_coarse_depth_for_mask = True"]),
  # ("013_bowl_01_novel_view", "013_bo01_nv_ms_exp17", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                           "NerfModel.use_predicted_mask = True",
  #                                                           "NerfModel.use_coarse_depth_for_mask = True",
  #                                                           "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
  #                                                           "NerfModel.use_x_in_rgb_condition = True"]),
  # ("011_bell_07_novel_view", "011_b07_nv_ms_exp17", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                           "NerfModel.use_predicted_mask = True",
  #                                                           "NerfModel.use_coarse_depth_for_mask = True",
  #                                                           "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
  #                                                           "NerfModel.use_x_in_rgb_condition = True"]),
  # ("014_spoon_02_novel_view", "014_s02_nv_ms_exp17", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                           "NerfModel.use_predicted_mask = True",
  #                                                           "NerfModel.use_coarse_depth_for_mask = True",
  #                                                           "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
  #                                                           "NerfModel.use_x_in_rgb_condition = True"]),

  # ("014_spoon_02_novel_view", "014_s02_nv_ms_exp19", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                           "NerfModel.use_predicted_mask = True",
  #                                                           "NerfModel.use_coarse_depth_for_mask = True",
  #                                                           "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
  #                                                           "NerfModel.use_x_in_rgb_condition = True",
  #                                                           "SpecularConfig.use_mask_occlusion_reg_loss = True",
  #                                                           "SpecularConfig.mask_occlusion_reg_loss_weight = 0.01"]),

  # ("014_spoon_02_novel_view", "014_s02_nv_ms_exp20", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                           "NerfModel.use_predicted_mask = True",
  #                                                           "NerfModel.use_coarse_depth_for_mask = True",
  #                                                           "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
  #                                                           "NerfModel.use_x_in_rgb_condition = True"]),
  # # gpu 3-2
  # ("014_spoon_02_novel_view", "014_s02_nv_ref_exp01", "ref", ["ExperimentConfig.image_scale = 1",
  #                                                             "NerfModel.use_x_in_rgb_condition = True"]),
  #
  # ("012_cup_01_novel_view", "012_c01_nv_ms_exp20", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                           "NerfModel.use_predicted_mask = True",
  #                                                           "NerfModel.use_coarse_depth_for_mask = True",
  #                                                           "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
  #                                                           "NerfModel.use_x_in_rgb_condition = True"]),
  # ("014_spoon_02_novel_view", "014_s02_nv_base_exp01", "base", ["ExperimentConfig.image_scale = 1"]),
  #
  # # gpu 3-3
  # ("013_bowl_01_novel_view", "013_bo01_nv_base_exp01", "base", ["ExperimentConfig.image_scale = 1"]),
  # ("013_bowl_01_novel_view", "013_bo01_nv_ms_exp20", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                           "NerfModel.use_predicted_mask = True",
  #                                                           "NerfModel.use_coarse_depth_for_mask = True",
  #                                                           "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
  #                                                           "NerfModel.use_x_in_rgb_condition = True"]),
  # ("011_bell_07_novel_view", "011_b07_nv_ms_exp20", "ms", ["ExperimentConfig.image_scale = 1",
  #                                                           "NerfModel.use_predicted_mask = True",
  #                                                           "NerfModel.use_coarse_depth_for_mask = True",
  #                                                           "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
  #                                                           "NerfModel.use_x_in_rgb_condition = True"]),

  # new
  # ("015_cup_02_novel_view", "015_c02_nv_ref_exp01", "ref", ["ExperimentConfig.image_scale = 1",
  #                                                             "NerfModel.use_x_in_rgb_condition = True"]),
  ("015_cup_02_novel_view", "015_c02_nv_ms_exp20", "ms", ["ExperimentConfig.image_scale = 1",
                                                            "NerfModel.use_predicted_mask = True",
                                                            "NerfModel.use_coarse_depth_for_mask = True",
                                                            "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                                                            "NerfModel.use_x_in_rgb_condition = True"]),
  ("016_spoon_03_novel_view", "016_s03_nv_ref_exp01", "ref", ["ExperimentConfig.image_scale = 1",
                                                              "NerfModel.use_x_in_rgb_condition = True"]),
  ("016_spoon_03_novel_view", "016_s03_nv_ms_exp20", "ms", ["ExperimentConfig.image_scale = 1",
                                                            "NerfModel.use_predicted_mask = True",
                                                            "NerfModel.use_coarse_depth_for_mask = True",
                                                            "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                                                            "NerfModel.use_x_in_rgb_condition = True"]),
]

def train_single(dataset_name, exp_name, config_key, gin_params, flow_exp_name):
  print("Start training for: {:20} {:15} {:10}".format(
    dataset_name, exp_name, config_key))

  dataset_path = os.path.join(data_root, dataset_name)
  exp_path = os.path.join(exp_root, exp_name)
  config_path = os.path.join(config_root, config_dict[config_key])
  flow_path = os.path.join(exp_root, flow_exp_name)

  process_str = [
    "python", "train.py",
    "--base_folder", exp_path,
    "--flow_folder", flow_path,
    "--gin_bindings=data_dir=\'{}\'".format(dataset_path),
    "--gin_configs", config_path
  ]
  for gin_param in gin_params:
    process_str += ["--gin_bindings", gin_param]

  pbar = tqdm(total=250000)
  with subprocess.Popen(process_str,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd="../",) as proc:
    while True:
      line = proc.stderr.readline()
      if not line:
        break
      line = line.decode('ascii').strip()
      print(line)
      match = re.match(r".*Saving checkpoint at step: ([\d]*)$", line)
      if match:
        checkpoint = int(match.group(1))
        pbar.update(checkpoint - pbar.n)
  pbar.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp_idx", nargs='+', type=int)
  args = parser.parse_args()

  training_schedule_selected = []
  for i, exp in enumerate(training_schedule):
    if not args.exp_idx or i in args.exp_idx:
      training_schedule_selected.append(exp)
  for training_setting in training_schedule_selected:
    try:
      gin_params = []
      flow_exp_name = ''
      if len(training_setting) == 3:
        dataset_name, exp_name, config_key = training_setting
      elif len(training_setting) == 4:
        dataset_name, exp_name, config_key, gin_params = training_setting
      elif len(training_setting) == 5:
        dataset_name, exp_name, config_key, gin_params, flow_exp_name = training_setting
      else:
        raise NotImplementedError
      train_single(dataset_name, exp_name, config_key, gin_params, flow_exp_name)
    except:
      print("Error encountered when running {}".format(exp_name))