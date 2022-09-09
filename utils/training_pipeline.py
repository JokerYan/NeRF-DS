import os
import re
import subprocess
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
  "hcxt": "test_local_spec_hcxt.gin",
  "vhcxt": "test_local_spec_vhcxt.gin",
}

data_root = ""
if os.path.isdir("/ssd/zhiwen/data/hypernerf/raw/"):
  data_root = "/ssd/zhiwen/data/hypernerf/raw/"
elif os.path.isdir("/hdd/zhiwen/data/hypernerf/raw/"):
  data_root = "/hdd/zhiwen/data/hypernerf/raw/"
elif os.path.isdir("/home/zwyan/3d_cv/data/hypernerf/raw/"):
  data_root = "/home/zwyan/3d_cv/data/hypernerf/raw/"

exp_root = "./experiments/"

# training schedule in the tuple of dataset_name, exp_name, config_key, gin_bindings
training_schedule = [
  # ("plate-1_qualitative", 'p1_q_hc_exp02', "hc"),
  # ("plate-1_qualitative", 'p1_q_ref_exp01', "ref"),
  # ("cup-2_qualitative", "c2_q_hc_exp01", "hc"),
  # ("cup-2_qualitative", "c2_q_ref_exp01", "ref"),

  # ("bell-3_qualitative", "b3_q_hcx_exp02", "hcx", ["SpecularConfig.use_hyper_c_jacobian_reg_loss=True"]),
  # ("bell-3_qualitative", "b3_q_hcx_exp01", "hcx", []),

  ("bell-2_qualitative", "b2_q_hcx_exp01", "hcx", []),
  # ("bell-3_qualitative", "b3_q_base_exp01", "base", []),

  # ("bell-2_qualitative", "b2_q_hc_exp01", "hc", []),
  # ("bell-2_qualitative", "b2_q_ref_exp01", "ref", []),

  # ("bell-1_qualitative", "b1_q_hc_exp03", "hc", ["NerfModel.stop_norm_gradient=False"]),  # delay w, no stop N
  # ("spoon-1_qualitative", "s1_q_hc_exp02", "hc", []),  # delay w
  # ("spoon-1_qualitative", "s1_q_hc_exp03", "hc", ["NerfModel.stop_norm_gradient=False"]),  # delay w, no stop N
]

def train_single(dataset_name, exp_name, config_key, gin_params):
  print("Start training for: {:20} {:15} {:10}".format(
    dataset_name, exp_name, config_key))

  dataset_path = os.path.join(data_root, dataset_name)
  exp_path = os.path.join(exp_root, exp_name)
  config_path = os.path.join(config_root, config_dict[config_key])

  process_str = [
    "python", "train.py",
    "--base_folder", exp_path,
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
      # print(line)
      match = re.match(r".*Saving checkpoint at step: ([\d]*)$", line)
      if match:
        checkpoint = int(match.group(1))
        pbar.update(checkpoint - pbar.n)
  pbar.close()


if __name__ == "__main__":
  for dataset_name, exp_name, config_key, gin_params in training_schedule:
    try:
      train_single(dataset_name, exp_name, config_key, gin_params)
    except:
      print("Error encountered when running {}".format(exp_name))