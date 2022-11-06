import argparse
from render import render_scene

interval = 1
# camera_path_name = 'fix_camera_93'
default_camera_path_name = 'vrig_camera'

# dataset_name, exp_prefix, config_key, exp_idx
render_schedule = [
  ("americano_masked", "am", "ms", "exp42", "fix_camera_1"),
  ("americano_masked", "am", "base", "exp02", "fix_camera_1"),

  ("011_bell_07_novel_view", "011_b07_nv", "base", "exp01"),
  ("015_cup_02_novel_view", "015_c02_nv", "base", "exp01"),
  ("018_as_01_novel_view", "018_a01_nv", "base", "exp01"),
  ("021_basin_01_novel_view", "021_bs01_nv", "base", "exp01"),
  ("022_sieve_02_novel_view", "022_sv02_nv", "base", "exp01"),
  ("025_press_01_novel_view", "025_ps01_nv", "base", "exp01"),
  ("026_bowl_02_novel_view", "026_bo02_nv", "base", "exp01"),
  ('028_plate_03_novel_view', '028_p03_nv', 'base', "exp01"),
  ("029_2cup_01_novel_view", "029_2c01_nv", "base", "exp01"),

  ("011_bell_07_novel_view", "011_b07_nv", "ref", "exp01"),
  ("015_cup_02_novel_view", "015_c02_nv", "ref", "exp01"),
  ("018_as_01_novel_view", "018_a01_nv", "ref", "exp01"),
  ("021_basin_01_novel_view", "021_bs01_nv", "ref", "exp01"),
  ("022_sieve_02_novel_view", "022_sv02_nv", "ref", "exp01"),
  ("025_press_01_novel_view", "025_ps01_nv", "ref", "exp01"),
  ("026_bowl_02_novel_view", "026_bo02_nv", "ref", "exp01"),
  ('028_plate_03_novel_view', '028_p03_nv', 'ref', "exp01"),
  ("029_2cup_01_novel_view", "029_2c01_nv", "ref", "exp01"),

  ("011_bell_07_novel_view", "011_b07_nv", "nerfies", "exp01"),
  ("015_cup_02_novel_view", "015_c02_nv", "nerfies", "exp01"),
  ("018_as_01_novel_view", "018_a01_nv", "nerfies", "exp01"),
  ("021_basin_01_novel_view", "021_bs01_nv", "nerfies", "exp01"),
  ("022_sieve_02_novel_view", "022_sv02_nv", "nerfies", "exp01"),
  ("025_press_01_novel_view", "025_ps01_nv", "nerfies", "exp01"),
  ("026_bowl_02_novel_view", "026_bo02_nv", "nerfies", "exp01"),
  ('028_plate_03_novel_view', '028_p03_nv', 'nerfies', "exp01"),
  ("029_2cup_01_novel_view", "029_2c01_nv", "nerfies", "exp01"),

  ("011_bell_07_novel_view", "011_b07_nv", "mso", "exp01"),
  ("015_cup_02_novel_view", "015_c02_nv", "mso", "exp01"),
  ("018_as_01_novel_view", "018_a01_nv", "mso", "exp01"),
  ("021_basin_01_novel_view", "021_bs01_nv", "mso", "exp01"),
  ("022_sieve_02_novel_view", "022_sv02_nv", "mso", "exp01"),
  ("025_press_01_novel_view", "025_ps01_nv", "mso", "exp01"),
  ("026_bowl_02_novel_view", "026_bo02_nv", "mso", "exp01"),
  ('028_plate_03_novel_view', '028_p03_nv', 'mso', "exp01"),
  ("029_2cup_01_novel_view", "029_2c01_nv", "mso", "exp01"),

  ("011_bell_07_novel_view", "011_b07_nv", "ms", "exp40"),
  ("015_cup_02_novel_view", "015_c02_nv", "ms", "exp40"),
  ("018_as_01_novel_view", "018_a01_nv", "ms", "exp40"),
  ("021_basin_01_novel_view", "021_bs01_nv", "ms", "exp40"),
  ("022_sieve_02_novel_view", "022_sv02_nv", "ms", "exp40"),
  ("025_press_01_novel_view", "025_ps01_nv", "ms", "exp40"),
  ("026_bowl_02_novel_view", "026_bo02_nv", "ms", "exp40"),
  ('028_plate_03_novel_view', '028_p03_nv', 'ms', "exp40"),
  ("029_2cup_01_novel_view", "029_2c01_nv", "ms", "exp40"),
]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp_idx", nargs='+', type=int)
  parser.add_argument("--exp_start", type=int)
  parser.add_argument("--exp_end", type=int)
  args = parser.parse_args()

  for i, scene in enumerate(render_schedule):
    if args.exp_idx is not None and i not in args.exp_idx:
      continue
    if args.exp_start is not None and not i >= args.exp_start:
      continue
    if args.exp_end is not None and not i < args.exp_end:
      continue

    if len(scene) == 4:
      dataset_name, exp_prefix, config_key, exp_idx = scene
      camera_path_name = default_camera_path_name
    elif len(scene) == 5:
      dataset_name, exp_prefix, config_key, exp_idx, camera_path_name = scene
    else:
      raise Exception
    exp_name = f'{exp_prefix}_{config_key}_{exp_idx}'
    print(f"rendering {dataset_name} {exp_name}")
    try:
      render_scene(dataset_name, exp_name, camera_path_name, interval)
    except Exception as e:
      print(e)
      print(f"Error rendering {exp_name}")

