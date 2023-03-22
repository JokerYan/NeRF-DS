import argparse
from render import render_scene

interval = 1
default_camera_path_name = 'vrig_camera'

# dataset_name, exp_prefix, config_key, exp_idx
render_schedule = [
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

    assert len(scene) == 3
    dataset_name, config_key, exp_idx = scene
    camera_path_name = default_camera_path_name
    exp_name = f'{dataset_name}_{config_key}_{exp_idx}'
    print(f"rendering {dataset_name} {exp_name}")
    try:
      render_scene(dataset_name, exp_name, camera_path_name, interval)
    except Exception as e:
      print(e)
      print(f"Error rendering {exp_name}")

