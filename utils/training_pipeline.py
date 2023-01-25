import os
import re
import subprocess
import argparse
from tqdm import tqdm
from pipeline_settings import pipeline_settings

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
  "bone": "test_local_spec_bone.gin",
  "nerfies": "test_local_spec_nerfies.gin",
  'mso': 'test_local_spec_mso.gin',
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
# if os.path.isdir('/ssd/zhiwen/exp/hypernerf'):
#   exp_root = '/ssd/zhiwen/exp/hypernerf'

# training schedule in the tuple of dataset_name, exp_prefix, config_key, exp_idx
training_schedule = [
  ("z-vrig-3dprinter", "z-vrig-3dprinter", "ms", "exp42"),
  ("z-vrig-broom", "z-vrig-broom", "ms", "exp42"),
  ("z-vrig-chicken", "z-vrig-chicken", "ms", "exp42"),
  ("z-vrig-peel-banana", "z-vrig-peel-banana", "ms", "exp42"),

  # ("018_as_01_novel_view", "018_a01_nv", "ms", "exp70"),
  # ("018_as_01_novel_view", "018_a01_nv", "ms", "exp71"),

  # ("018_as_01_novel_view", "018_a01_nv", "ms", "exp60"),
  # ("018_as_01_novel_view", "018_a01_nv", "ms", "exp61"),
  # ("018_as_01_novel_view", "018_a01_nv", "ms", "exp62"),
  # ("018_as_01_novel_view", "018_a01_nv", "ms", "exp63"),

  # ("018_as_01_novel_view", "018_a01_nv", "ms", "exp50"),
  # ("018_as_01_novel_view", "018_a01_nv", "ms", "exp51"),
  # ("018_as_01_novel_view", "018_a01_nv", "ms", "exp52"),
  # ("018_as_01_novel_view", "018_a01_nv", "ms", "exp53"),
  # ("018_as_01_novel_view", "018_a01_nv", "ms", "exp54"),

  # ("021_basin_01_novel_view", "021_bs01_nv", "ms", "exp50"),
  # ("021_basin_01_novel_view", "021_bs01_nv", "ms", "exp51"),
  # ("021_basin_01_novel_view", "021_bs01_nv", "ms", "exp52"),
  # ("021_basin_01_novel_view", "021_bs01_nv", "ms", "exp53"),
  # ("021_basin_01_novel_view", "021_bs01_nv", "ms", "exp54"),

  # ("americano_masked", "am", "base", "exp03"),
  # ("011_bell_07_novel_view", "011_b07_nv", "ms", "exp50"),
  # ("015_cup_02_novel_view", "015_c02_nv", "ms", "exp50"),
  #
  # ("025_press_01_novel_view", "025_ps01_nv", "ms", "exp41"),

  # ("americano_masked", "am", "ms", "exp42"),
  # ("americano_masked", "am", "base", "exp02"),

  # ("021_basin_01_novel_view", "021_bs01_nv", "ms", "exp40"),
  # ("026_bowl_02_novel_view", "026_bo02_nv", "nerfies", "exp01"),
  # ('028_plate_03_novel_view', '028_p03_nv', 'nerfies', "exp01"),
  #
  # ("011_bell_07_novel_view", "011_b07_nv", "mso", "exp01"),
  # ("015_cup_02_novel_view", "015_c02_nv", "mso", "exp01"),
  # ("018_as_01_novel_view", "018_a01_nv", "mso", "exp01"),
  # ("021_basin_01_novel_view", "021_bs01_nv", "mso", "exp01"),
  # ("022_sieve_02_novel_view", "022_sv02_nv", "mso", "exp01"),
  # ("025_press_01_novel_view", "025_ps01_nv", "mso", "exp01"),
  # ("026_bowl_02_novel_view", "026_bo02_nv", "mso", "exp01"),
  # ('028_plate_03_novel_view', '028_p03_nv', 'mso', "exp01"),
  # ("029_2cup_01_novel_view", "029_2c01_nv", "mso", "exp01"),

  # ("022_sieve_02_novel_view", "022_sv02_nv", "nerfies", "exp01"),
  # ("025_press_01_novel_view", "025_ps01_nv", "nerfies", "exp01"),
  # ("026_bowl_02_novel_view", "026_bo02_nv", "nerfies", "exp01"),
  # ('028_plate_03_novel_view', '028_p03_nv', 'nerfies', "exp01"),

  # ("011_bell_07_novel_view", "011_b07_nv", "ms", "exp40"),
  # ("015_cup_02_novel_view", "015_c02_nv", "ms", "exp40"),
  # ("026_bowl_02_novel_view", "026_bo02_nv", "base", "exp01"),

  # ("022_sieve_02_novel_view", "022_sv02_nv", "ms", "exp40"),
  # ("022_sieve_02_novel_view", "022_sv02_nv", "ref", "exp01"),

  # ("025_press_01_novel_view", "025_ps01_nv", "base", "exp01"),

  # ("023_tray_01_novel_view", "023_tr01_nv", "ref", "exp01"),
  # ("023_tray_01_novel_view", "023_tr01_nv", "ms", "exp40"),

  # gpu server
  # ("029_2cup_01_novel_view", "029_2c01_nv", "base", "exp01"),
  # ("029_2cup_01_novel_view", "029_2c01_nv", "nerfies", "exp01"),

  # gpu server
  # ("029_2cup_01_novel_view", "029_2c01_nv", "ms", "exp40"),
  # ("029_2cup_01_novel_view", "029_2c01_nv", "ref", "exp01"),
  #
  # ('028_plate_03_novel_view', '028_p03_nv', 'base', "exp01"),

  # gpu server
  # ('016_spoon_03_novel_view', '016_s03_nv', "ms", "exp40"),
  # ('018_as_01_novel_view', '018_a01_nv', "ms", "exp40"),
  #
  # ('027_dryer_01_novel_view', '027_dr01_nv', "ms", "exp40"),
  # ('027_dryer_01_novel_view', '027_dr01_nv', "ref", "exp01"),
  #
  # ('028_plate_03_novel_view', '028_p03_nv', 'ms', "exp40"),
  # ('028_plate_03_novel_view', '028_p03_nv', 'ref', "exp01"),

  # gpu server
  # ("026_bowl_02_novel_view", "026_bo02_nv", "ms", "exp40"),
  # ("026_bowl_02_novel_view", "026_bo02_nv", "ref", "exp01"),
  #
  # ("025_press_01_novel_view", "025_ps01_nv", "base", "exp01"),
  # ("022_sieve_02_novel_view", "022_sv02_nv", "base", "exp01"),

  # ("025_press_01_novel_view", "025_ps01_nv", "ms", "exp40"),
  # ("025_press_01_novel_view", "025_ps01_nv", "ref", "exp01"),
  #
  # ("024_box_01_novel_view", "024_bx01_nv", "ms", "exp40"),
  # ("024_box_01_novel_view", "024_bx01_nv", "ref", "exp01"),


  # ('022_sieve_02_novel_view', '022_sv02_nv', 'ms', 'exp39')

  # finished
  # ('018_as_01_novel_view', '018_a01_nv', "nerfies", "exp01"),
  # ("021_basin_01_novel_view", "021_bs01_nv", "nerfies", "exp01"),
  # ("011_bell_07_novel_view", "011_b07_nv", "ms", "exp39"),
  # ("015_cup_02_novel_view", "015_c02_nv", "ms", "exp39"),

  # not finished
  # ('018_as_01_novel_view', '018_a01_nv', "ms", "exp39"),
  # ("021_basin_01_novel_view", "021_bs01_nv", "ms", "exp39"),

  # finished
  # ("011_bell_07_novel_view", "011_b07_nv", "ms", "exp34"),
  # ("011_bell_07_novel_view", "011_b07_nv", "ms", "exp35"),
  # ("011_bell_07_novel_view", "011_b07_nv", "ms", "exp31"),

  # ("015_cup_02_novel_view", "015_c02_nv", "ms", "exp34"),
  # ("017_cup_03_novel_view", "017_c03_nv", "ms", "exp34"),

  # ("014_spoon_02_novel_view", "014_s02_nv", "base", "exp01"),
  # ("014_spoon_02_novel_view", "014_s02_nv", "ref", "exp01"),

  # ("011_bell_07_novel_view", "011_b07_nv", "ms", "exp36"),

  # gpu server
  # ("020_sieve_01_novel_view", "020_sv01_nv", "ms", "exp39"),
  # ("020_sieve_01_novel_view", "020_sv01_nv", "ms", "exp38"),
  #
  # ("020_sieve_01_novel_view", "020_sv01_nv", "base", "exp01"),
  # ("021_basin_01_novel_view", "021_bs01_nv", "base", "exp01"),

  # ('017_cup_03_novel_view', '017_c03_nv', "ms", "exp39"),
  # ('019_plate_01_novel_view', '019_p01_nv', "ms", "exp39"),

  #
  # # gpu server
  # ("020_sieve_01_novel_view", "020_sv01_nv", "ms", "exp36"),
  # ("020_sieve_01_novel_view", "020_sv01_nv", "ref", "exp01"),
  #
  # ("021_basin_01_novel_view", "021_bs01_nv", "ms", "exp36"),
  # ("021_basin_01_novel_view", "021_bs01_nv", "ref", "exp01"),
  #
  # # gpu server
  # ("011_bell_07_novel_view", "011_b07_nv", "nerfies", "exp01"),
  # ("015_cup_02_novel_view", "015_c02_nv", "nerfies", "exp01"),

  # gpu server
  # ("014_spoon_02_novel_view", "014_s02_nv", "ms", "exp36"),
  # ("015_cup_02_novel_view", "015_c02_nv", "ms", "exp36"),

  # gpu server
  # ('018_as_01_novel_view', '018_a01_nv', "ref", "exp01"),
  # ('018_as_01_novel_view', '018_a01_nv', "ms", "exp36"),
  #
  # ('019_plate_01_novel_view', '019_p01_nv', "ref", "exp01"),
  # ('019_plate_01_novel_view', '019_p01_nv', "ms", "exp36"),


  # gpu server
  # ('018_as_01_novel_view', '018_a01_nv', "ms", "exp35"),
  # ('019_plate_01_novel_view', '019_p01_nv', "ms", "exp35"),

  # ('018_as_01_novel_view', '018_a01_nv', "base", "exp01"),
  # ('018_as_01_novel_view', '018_a01_nv', "ref", "exp01"),
  #
  # ('019_plate_01_novel_view', '019_p01_nv', "base", "exp01"),
  # ('019_plate_01_novel_view', '019_p01_nv', "ref", "exp01"),
  #
  # ('018_as_01_novel_view', '018_a01_nv', "ms", "exp29"),
  # ('019_plate_01_novel_view', '019_p01_nv', "ms", "exp29"),

  # # gpu server
  # ("015_cup_02_novel_view", "015_c02_nv", "ms", "exp32"),
  # ("015_cup_02_novel_view", "015_c02_nv", "ms", "exp31"),
  #
  # ("017_cup_03_novel_view", "017_c03_nv", "ms", "exp32"),
  # ("017_cup_03_novel_view", "017_c03_nv", "ms", "exp31"),
  #
  # ("017_cup_03_novel_view", "017_c03_nv", "ref", "exp01"),
  # ("017_cup_03_novel_view", "017_c03_nv", "base", "exp01"),
  # ("015_cup_02_novel_view", "015_c02_nv", "ms", "exp33"),

  # wait rendering
  # ("015_cup_02_novel_view", "015_c02_nv", "ms", "exp30"),
  # ("014_spoon_02_novel_view", "014_s02_nv", "ms", "exp30"),

  # bone
  # ("011_bell_07_novel_view", "011_b07_nv", "bone", "exp01"),
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
      dataset_name, exp_prefix, config_key, exp_idx = training_setting
      exp_name = f'{exp_prefix}_{config_key}_{exp_idx}'
      gin_params = pipeline_settings[config_key][exp_idx]
      # if len(training_setting) == 3:
      #   dataset_name, exp_name, config_key = training_setting
      # elif len(training_setting) == 4:
      #   dataset_name, exp_name, config_key, gin_params = training_setting
      # elif len(training_setting) == 5:
      #   dataset_name, exp_name, config_key, gin_params, flow_exp_name = training_setting
      # else:
      #   raise NotImplementedError
      train_single(dataset_name, exp_name, config_key, gin_params, flow_exp_name)
    except Exception as e:
      print(e)
      print("Error encountered when running {}".format(exp_name))