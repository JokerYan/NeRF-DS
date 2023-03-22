import os
import re
import subprocess
import argparse
from tqdm import tqdm
from pipeline_settings import pipeline_settings

config_root = "./configs/"
config_dict = {
  'base': "test_base.gin",
  "ds": "nerf_ds.gin",
}

silent = False
data_root = ""
if os.path.isdir("/ssd/zhiwen/data/hypernerf/raw/"):
  data_root = "/ssd/zhiwen/data/hypernerf/raw/"
elif os.path.isdir("/hdd/zhiwen/data/hypernerf/raw/"):
  data_root = "/hdd/zhiwen/data/hypernerf/raw/"
elif os.path.isdir("/home/zwyan/3d_cv/data/hypernerf/raw/"):
  data_root = "/home/zwyan/3d_cv/data/hypernerf/raw/"

exp_root = "./experiments/"

# training schedule in the tuple of dataset_name, exp_prefix, config_key, exp_idx
training_schedule = [
  ("bell_novel_view", "ds", "exp01"),
  ("cup_novel_view", "ds", "exp01"),
  ("as_novel_view", "ds", "exp01"),
  ("basin_novel_view", "ds", "exp01"),
  ("sieve_novel_view", "ds", "exp01"),
  ("press_novel_view", "ds", "exp01"),
  ("bowl_novel_view", "ds", "exp01"),
  ('plate_novel_view', 'ds', "exp01"),
  ("2cup_novel_view", "ds", "exp01"),
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

  total_steps = 2500000 if exp_name.endswith('exp44') else 250000
  pbar = tqdm(total=total_steps)
  with subprocess.Popen(process_str,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd="../",) as proc:
    while True:
      line = proc.stderr.readline()
      if not line:
        break
      line = line.decode('ascii').strip()
      if not silent:
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
      flow_exp_name = ''
      dataset_name, config_key, exp_idx = training_setting
      exp_name = f'{dataset_name}_{config_key}_{exp_idx}'
      gin_params = pipeline_settings[config_key][exp_idx]
      train_single(dataset_name, exp_name, config_key, gin_params, flow_exp_name)
    except Exception as e:
      print(e)
      print("Error encountered when running {}".format(exp_name))