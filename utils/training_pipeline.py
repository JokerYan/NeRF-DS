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
  # ("plate-1_qualitative", 'p1_q_hc_exp02', "hc"),
  # ("plate-1_qualitative", 'p1_q_ref_exp01', "ref"),
  # ("cup-2_qualitative", "c2_q_hc_exp01", "hc"),
  # ("cup-2_qualitative", "c2_q_ref_exp01", "ref"),

  # ("bell-3_qualitative", "b3_q_hcx_exp02", "hcx", ["SpecularConfig.use_hyper_c_jacobian_reg_loss=True"]),
  # ("bell-3_qualitative", "b3_q_hcx_exp01", "hcx", []),

  # ("bell-2_qualitative", "b2_q_hcx_exp01", "hcx", []),
  # ("bell-3_qualitative", "b3_q_base_exp01", "base", []),

  # ("bell-2_qualitative", "b2_q_hc_exp01", "hc", []),
  # ("bell-2_qualitative", "b2_q_ref_exp01", "ref", []),

  # ("bell-1_qualitative", "b1_q_hc_exp03", "hc", ["NerfModel.stop_norm_gradient=False"]),  # delay w, no stop N
  # ("spoon-1_qualitative", "s1_q_hc_exp02", "hc", []),  # delay w
  # ("spoon-1_qualitative", "s1_q_hc_exp03", "hc", ["NerfModel.stop_norm_gradient=False"]),  # delay w, no stop N

  # ("bell-2_qualitative", "b2_q_hcx_nv_exp02", "hcx_nv", []),
  # ("bell-2_qualitative", "b2_q_hcx_nv_exp03", "hcx_nv", ["NerfModel.hyper_c_num_dims = 4"]),
  # ("bell-3_qualitative", "b3_q_hcx_nv_exp04", "hcx_nv", ["NerfModel.hyper_c_num_dims = 4",
  #                                                        "NerfModel.use_ref_radiance = True"]),

  # ("bell-3_qualitative", "b3_q_hcx_nv_exp05", "hcx_nv", ["NerfModel.use_ref_radiance = True"]),
  # ("bell-3_qualitative", "b3_q_hcx_nv_exp06", "hcx_nv", ["NerfModel.use_ref_radiance = True"]),
  # ("bell-2_qualitative", "b2_q_hcx_nv_exp06", "hcx_nv", ["NerfModel.use_ref_radiance = True"]),

  # ("spoon-1_qualitative", "s1_q_hcx_vn_exp01", "hcx_nv", []),

  # ("bell-2_qualitative", "b2_q_hs_exp02", "hs", []),
  # ("bell-2_qualitative", "b2_q_hs_exp03", "hs", ["HyperSpecModel.use_hyper_c = False",
  #                                                ]),
  # ("bell-2_qualitative", "b2_q_hs_exp04", "hs", ["HyperSpecModel.nerf_rgb_branch_depth = 3",
  #                                                ]),
  # ("bell-2_qualitative", "b2_q_hs_exp05", "hs", ["HyperSpecModel.use_norm_voxel = False",
  #                                                ])

  # ("vrig-bell-1_multi_view", "vb1_nv_hcx_mv_exp01", "vhs", ["ExperimentConfig.image_scale = 1"]),       # local
  # ("vrig-bell-1_multi_view", "vb1_nv_hcx_mv_exp02", "vhs", ["ExperimentConfig.image_scale = 1",
  #                                                           "HyperSpecModel.use_hyper_c = False"]),       # local

  # ("vrig-bell-1_multi_view", "vb1_nv_hcx_mv_exp03", "vhs", ["ExperimentConfig.image_scale = 1",
  #                                                           "HyperSpecModel.nerf_rgb_branch_depth = 3"]),       # local

  # ("vrig-bell-1_novel_view", "vb1_nv_hs_exp05", "vhs", ["ExperimentConfig.image_scale = 1",
  #                                                       "HyperSpecModel.use_norm_voxel = False"]),
  # ("vrig-bell-1_novel_view", "vb1_nv_hs_exp06", "vhs", ["ExperimentConfig.image_scale = 1",
  #                                                       "HyperSpecModel.use_norm_voxel = False",
  #                                                       "HyperSpecModel.use_hyper = False",
  #                                                       "TrainConfig.use_elastic_loss = True"]),

  # ("vrig-cup-3_qualitative", "vc3_q_hs_exp02", "hs", ["ExperimentConfig.image_scale = 1",
  #                                                     "TrainConfig.early_stop_steps = 90000"]),  # local
  # ("vrig-cup-3_qualitative", "vc3_q_base_exp01", "base", ["ExperimentConfig.image_scale = 1",
  #                                                         "TrainConfig.early_stop_steps = 90000"]),  # local
  # ("vrig-cup-3_qualitative", "vc3_q_ref_exp01", "ref", ["ExperimentConfig.image_scale = 1",
  #                                                       "TrainConfig.early_stop_steps = 90000"]),  # local


  # ("vrig-bell-1_novel_view", "vb1_nv_hcx_nv_exp01", "vhcx_nv", ["ExperimentConfig.image_scale = 1"]),       # local

  # ("vrig-bell-1_novel_view", "vb1_nv_hs_exp02", "vhs", ["ExperimentConfig.image_scale = 1"]),       # local
  # ("vrig-bell-1_novel_view", "vb1_nv_ref_exp01", "vref", ["ExperimentConfig.image_scale = 1"]),     # gpu3-2
  # ("vrig-bell-1_novel_view", "vb1_nv_base_exp01", "vbase", ["ExperimentConfig.image_scale = 1"]),   # gpu3-3
  # ("bell-3_qualitative", "b3_q_hs_exp02", "hs", []),                                                # gpu3-1

  # ("plate-1_qualitative", "p1_q_hs_exp02", "hs", []),

  # ("hypernerf_americano", "h_am_q_base_exp01", "base", []),
  # ("hypernerf_cross-hands1", "h_ch1_q_base_exp01", "base", []),
  # ("hypernerf_keyboard", "h_k_q_base_exp01", "base", []),

  # ("bell-2_qualitative", "b2_q_hsf_exp01", "hsf", [], "b2_q_ref_exp01"),
  # ("bell-2_qualitative", "b2_q_hsf_exp02", "hsf", [], "b2_q_ref_exp01"),

  # ("bell-2_qualitative", "b2_q_hsf_exp05", "hsf", ["HyperSpecModel.use_norm_voxel = False"], "b2_q_ref_exp01"),
  # ("plate-1_qualitative", "p1_q_hsf_exp05", "hsf", ["HyperSpecModel.use_norm_voxel = False"], "p1_q_ref_exp01"),
  # ("bell-2_qualitative", "b2_q_hsf_exp06", "hsf", [], "b2_q_ref_exp01"),
  # ("vrig-cup-3_qualitative", "vc3_q_hsf_exp06", "hsf", ["ExperimentConfig.image_scale = 1"], "vc3_q_ref_exp01"),
  # ("vrig-cup-3_qualitative", "vc3_q_hsf_exp05", "hsf", ["ExperimentConfig.image_scale = 1",
  #                                                       "flow_model_light_learning_rate_sched = {'type': 'constant','value': 0,}"
  #                                                       ], "vc3_q_ref_exp01"),

  # ("bell-3_qualitative", "b3_q_ref_exp02", "ref", []),
  # ("bell-3_qualitative", "b3_q_base_exp02", "base", []),
  # ("bell-3_qualitative", "b3_q_hsf_exp06", "hsf", [], "b3_q_ref_exp02"),
  # ("bell-3_qualitative", "b3_q_hsf_exp07", "hsf", ["HyperSpecModel.norm_supervision_type = 'warped'"], "b3_q_ref_exp02"),

  # ("000_bell_01_novel_view", "000_b01_nv_ref_exp01", "ref", ["ExperimentConfig.image_scale = 1"]),
  # ("002_bell_03_novel_view", "002_b03_nv_ref_exp01", "ref", ["ExperimentConfig.image_scale = 1"]),
  # ("006_spoon_01_novel_view", "006_s01_nv_ref_exp01", "ref", ["ExperimentConfig.image_scale = 1"]),
  # ("008_bell_04_novel_view", "008_b04_nv_ref_exp01", "ref", ["ExperimentConfig.image_scale = 1"]),
  # ("009_bell_05_novel_view", "009_b05_nv_ref_exp01", "ref", ["ExperimentConfig.image_scale = 1"]),

  # ("008_bell_04_novel_view", "008_b04_nv_hsf_exp01", "hsf", ["ExperimentConfig.image_scale = 1"], "008_b04_nv_ref_exp01"),
  # ("009_bell_05_novel_view", "009_b05_nv_hsf_exp01", "hsf", ["ExperimentConfig.image_scale = 1"], "009_b05_nv_ref_exp01"),

  # ("010_bell_06_novel_view", "010_b06_nv_ref_exp01", "ref", ["ExperimentConfig.image_scale = 1"]),
  # ("010_bell_06_novel_view", "010_b06_nv_hsf_exp01", "hsf", ["ExperimentConfig.image_scale = 1"], "010_b06_nv_ref_exp01"),

  # ("011_bell_07_novel_view", "011_b07_nv_ref_exp01", "ref", ["ExperimentConfig.image_scale = 1"]),
  # ("012_cup_01_novel_view", "012_c01_nv_ref_exp01", "ref", ["ExperimentConfig.image_scale = 1"]),
  # ("011_bell_07_novel_view", "011_b07_nv_hsf_exp01", "hsf", ["ExperimentConfig.image_scale = 1"], "011_b07_nv_ref_exp01"),
  # ("012_cup_01_novel_view", "012_c01_nv_hsf_exp01", "hsf", ["ExperimentConfig.image_scale = 1"], "012_c01_nv_ref_exp01"),

  # ("011_bell_07_novel_view", "011_b07_nv_ms_exp02", "ms", ["ExperimentConfig.image_scale = 1"]),
  # ("012_cup_01_novel_view", "012_c01_nv_ms_exp02", "ms", ["ExperimentConfig.image_scale = 1"]),

  ("011_bell_07_novel_view", "011_b07_nv_ms_exp03", "ms", ["ExperimentConfig.image_scale = 1",
                                                           "SpecularConfig.use_mask_weighted_loss = False",
                                                           "SpecularConfig.use_mask_consistency_loss = True"]),
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