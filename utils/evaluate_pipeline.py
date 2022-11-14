import os
import time
import numpy as np
import pickle

from data_abbreviations import data_abbr
from load_results import load_gt, load_hypernerf, load_refnerf
from calculate_quantitative_results import calculate as calculate_quantitative


def evaluate_single(dataset_name, config_key, exp_idx=''):
  print(f"==> Evaluating {dataset_name} {config_key} {exp_idx}")
  is_refnerf = config_key == 'refnerf'
  dataset_name_nv = f'{dataset_name}_novel_view'

  gt_images = load_gt(dataset_name_nv)
  if is_refnerf:
    out_images = load_refnerf(dataset_name)
  else:
    exp_prefix = data_abbr[dataset_name] + '_nv'
    out_images = load_hypernerf(exp_prefix, config_key, exp_idx)

  mse_list, psnr_list, ms_ssim_list, lpips_list = calculate_quantitative(gt_images, out_images)
  mse = np.mean(mse_list)
  psnr = np.mean(psnr_list)
  ms_ssim = np.mean(ms_ssim_list)
  lpips = np.mean(lpips_list)
  result_str = "mse: {:.5f} psnr: {:.3f} ms_ssim: {:.3f} lpips: {:.3f}".format(
    mse, psnr, ms_ssim, lpips)
  print(result_str)
  return mse, psnr, ms_ssim, lpips


dataset_pipeline = [
  "011_bell_07",
  "015_cup_02",
  "018_as_01",
  "021_basin_01",
  "022_sieve_02",
  "025_press_01",
  "026_bowl_02",
  "028_plate_03",
  "029_2cup_01",
]
exp_pipeline = [
  ("ms", "exp40"),
  ("ref", "exp01"),
  ("mso", "exp01"),
  ("base", "exp01"),
  ("nerfies", "exp01"),
  ("refnerf", ""),
]
out_dir = '/home/zwyan/3d_cv/repos/hypernerf_barf/evaluations/'
def evaluate_pipeline():
  # permutation
  evaluate_setups = []
  for dataset_name in dataset_pipeline:
    for config_key, exp_idx in exp_pipeline:
      evaluate_setups.append((dataset_name, config_key, exp_idx))

  # evaluate
  evaluate_result_dict = {}
  for setup in evaluate_setups:
    dataset_name, config_key, exp_idx = setup
    mse, psnr, ms_ssim, lpips = evaluate_single(dataset_name, config_key, exp_idx)
    evaluate_result_dict[setup] = {
      'mse': mse,
      'psnr': psnr,
      'ms_ssim': ms_ssim,
      'lpips': lpips
    }

  # save
  out_name = f'evaluation_{time.time()}.pkl'
  out_path = os.path.join(out_dir, out_name)
  # np.save(out_path, evaluate_result_dict)
  with open(out_path, 'wb') as f:
    pickle.dump(evaluate_result_dict, f)
  print(f"result saved to :{out_path}")


selected_eval_methods = ['ms_ssim', "psnr", "lpips"]
selected_eval_string = ['{:.3f}', '{:.1f}', '{:.3f}']
def save_npy_to_csv(pickle_name):
  pickle_path = os.path.join(out_dir, pickle_name)
  result = np.load(pickle_path, allow_pickle=True)

  dataset_row = ['']
  eval_method_row = ['']
  result_rows = [[f'{config_key}_{exp_idx}'] for config_key, exp_idx in exp_pipeline]

  for dataset in dataset_pipeline:
    dataset_row += [dataset] + ['' for i in range(len(selected_eval_methods) - 1)]
    eval_method_row += selected_eval_methods
    for i, (config_key, exp_idx) in enumerate(exp_pipeline):
      exp_result = result[(dataset, config_key, exp_idx)]
      exp_result_list = [exp_result[eval_method] for eval_method in selected_eval_methods]
      exp_result_str_list = [selected_eval_string[j].format(exp_result_list[j]) for j in range(len(selected_eval_methods))]
      result_rows[i] += exp_result_str_list
  # combine all
  csv_list = []
  csv_list.append(dataset_row)
  csv_list.append(eval_method_row)
  csv_list += result_rows
  csv_str = '\n'.join([','.join(row) for row in csv_list])

  assert pickle_name.endswith('.pkl')
  csv_name = pickle_name[:-4] + '.csv'
  csv_path = os.path.join(out_dir, csv_name)
  with open(csv_path, 'w+') as f:
    f.write(csv_str)
  print(f'csv saved to: {csv_path}')


if __name__ == "__main__":
  # evaluate_single("011_bell_07", "base", "exp01")
  # evaluate_single("011_bell_07", "refnerf")
  # evaluate_pipeline()
  save_npy_to_csv("evaluation_1667801066.6375782.pkl")