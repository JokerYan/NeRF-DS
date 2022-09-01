import os
import re
import subprocess
from tqdm import tqdm

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
  "vhc": "test_local_spec_hc_vrig.gin"
}

data_root = ""
if os.path.isdir("/ssd/zhiwen/data/hypernerf/raw/"):
  data_root = "/ssd/zhiwen/data/hypernerf/raw/"
elif os.path.isdir("/hdd/zhiwen/data/hypernerf/raw/"):
  data_root = "/hdd/zhiwen/data/hypernerf/raw/"
elif os.path.isdir("/home/zwyan/3d_cv/data/hypernerf/raw/"):
  data_root = "/home/zwyan/3d_cv/data/hypernerf/raw/"

exp_root = "./experiments/"

# training schedule in the tuple of dataset_name, exp_name, config_key
training_schedule = [
  ("white-board-5", "wb5_exp03", "ref"),
  # ("aluminium-sheet-6", "as6_exp03_base", "base"),
  # ("vrig-white-board-1_qualitative", "vwb1_q_hc_exp01", "hc"),
  # ("vrig-white-board-1_novel_view", "vwb1_nv_hc_exp02", "vhc"),
  # ("vrig-white-board-1_novel_view", "vwb1_nv_ref_exp01", "vref")
]

# training_schedule = [
#   ("vrig-white-board-1_novel_view", "vwb1_nv_hc_exp01", "vhc")
# ]

def train_single(dataset_name, exp_name, config_key):
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
  for dataset_name, exp_name, config_key in training_schedule:
    train_single(dataset_name, exp_name, config_key)