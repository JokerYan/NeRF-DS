from render import render_scene

interval = 9
# camera_path_name = 'fix_camera_93'
camera_path_name = 'vrig_camera'

# dataset_name, exp_prefix, config_key, exp_idx
render_schedule = [
  # ("011_bell_07_novel_view", "011_b07_nv", "ms", "exp36"),
  # ("011_bell_07_novel_view", "011_b07_nv", "ref", "exp01"),
  # ("011_bell_07_novel_view", "011_b07_nv", "base", "exp01"),
  #
  # ("015_cup_02_novel_view", "015_c02_nv", "ms", "exp40"),
  # ("015_cup_02_novel_view", "015_c02_nv", "ref", "exp01"),
  # ("015_cup_02_novel_view", "015_c02_nv", "base", "exp01"),
  #
  # ('018_as_01_novel_view', '018_a01_nv', "ms", "exp36"),
  # ('018_as_01_novel_view', '018_a01_nv', "ref", "exp01"),
  # ('018_as_01_novel_view', '018_a01_nv', "base", "exp01"),

  ("021_basin_01_novel_view", "021_bs01_nv", "ms", "exp36"),
  # ("021_basin_01_novel_view", "021_bs01_nv", "ref", "exp01"),
  # ("021_basin_01_novel_view", "021_bs01_nv", "base", "exp01"),
]

if __name__ == "__main__":
  for scene in render_schedule:
    dataset_name, exp_prefix, config_key, exp_idx = scene
    exp_name = f'{exp_prefix}_{config_key}_{exp_idx}'
    print(f"rendering {dataset_name} {exp_name}")
    try:
      render_scene(dataset_name, exp_name, camera_path_name, interval)
    except Exception as e:
      print(e)
      print(f"Error rendering {exp_name}")

