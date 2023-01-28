import glob
import os
import cv2

import jax
import jax.numpy as jnp
import torch
from jax import random

import flax
from flax import jax_utils
from flax import optim
from flax.training import checkpoints

import functools
from absl import logging
import numpy as np
import mediapy
from pathlib import Path
from jax import tree_util
import gin
from IPython.display import display, Markdown
from tqdm import tqdm
import time
import math

from hypernerf import evaluation
from hypernerf import schedules
from hypernerf import training
from hypernerf import models
from hypernerf import configs
from hypernerf import datasets
from hypernerf import image_utils
from hypernerf import visualization as viz
from hypernerf import model_utils
from hypernerf import utils


chunk_size = 2048
def render_scene(dataset_name, exp_name, camera_path_name, interval, norm_override=None):
  # print('Detected Devices:', jax.devices())

  # @title Define imports and utility functions.
  # Monkey patch logging.
  def myprint(msg, *args, **kwargs):
    pass
    # print(msg % args)

  logging.info = myprint
  logging.warn = myprint
  logging.error = myprint
  # @title Model and dataset configuration
  # @markdown Change the directories to where you saved your capture and experiment.
  if os.path.exists('/ssd/zhiwen/data/hypernerf/raw/'):
    data_root = '/ssd/zhiwen/data/hypernerf/raw/'
    project_root = '/data/zwyan/hypernerf-barf'
  elif os.path.exists('/hdd/zhiwen/data/hypernerf/raw/'):
    data_root = '/hdd/zhiwen/data/hypernerf/raw/'
    project_root = '/data/zwyan/hypernerf-barf'
  elif os.path.exists('/home/zwyan/3d_cv/data/hypernerf/raw/'):
    data_root = '/home/zwyan/3d_cv/data/hypernerf/raw/'
    project_root = '/home/zwyan/3d_cv/repos/hypernerf_barf'
  else:
    raise NotImplemented

  # @markdown The working directory where the trained model is.
  train_dir = os.path.join(project_root, 'experiments', exp_name)

  # @markdown The directory to the dataset capture.
  data_dir = os.path.join(data_root, dataset_name)  # @param {type: "string"}

  checkpoint_dir = Path(train_dir, 'checkpoints')
  checkpoint_dir.mkdir(exist_ok=True, parents=True)
  config_path = Path(train_dir, 'config.gin')
  with open(config_path, 'r') as f:
    logging.info('Loading config from %s', config_path)
    config_str = f.read()
  gin.parse_config(config_str)

  exp_config = configs.ExperimentConfig()
  train_config = configs.TrainConfig()
  eval_config = configs.EvalConfig()
  spec_config = configs.SpecularConfig()
  display(Markdown(
    gin.config.markdown(gin.config_str())))

  # @title Create datasource and show an example.
  if spec_config.use_hyper_spec_model:
    dummy_model = models.HyperSpecModel({}, 0, 0)
  else:
    dummy_model = models.NerfModel({}, 0, 0)
  datasource = exp_config.datasource_cls(
    data_dir=data_dir,
    image_scale=exp_config.image_scale,
    random_seed=exp_config.random_seed,
    # Enable metadata based on model needs.
    use_warp_id=dummy_model.use_warp,
    use_appearance_id=(
            dummy_model.nerf_embed_key == 'appearance'
            or dummy_model.hyper_embed_key == 'appearance'),
    use_camera_id=dummy_model.nerf_embed_key == 'camera',
    use_time=dummy_model.warp_embed_key == 'time')

  # @title Load model
  # @markdown Defines the model and initializes its parameters.
  rng = random.PRNGKey(exp_config.random_seed)
  np.random.seed(exp_config.random_seed + jax.process_index())
  devices_to_use = jax.devices()

  learning_rate_sched = schedules.from_config(train_config.lr_schedule)
  nerf_alpha_sched = schedules.from_config(train_config.nerf_alpha_schedule)
  warp_alpha_sched = schedules.from_config(train_config.warp_alpha_schedule)
  elastic_loss_weight_sched = schedules.from_config(
    train_config.elastic_loss_weight_schedule)
  hyper_alpha_sched = schedules.from_config(train_config.hyper_alpha_schedule)
  hyper_sheet_alpha_sched = schedules.from_config(
    train_config.hyper_sheet_alpha_schedule)
  norm_loss_weight_sched = schedules.from_config(spec_config.norm_loss_weight_schedule)
  norm_input_alpha_sched = schedules.from_config(spec_config.norm_input_alpha_schedule)
  norm_voxel_lr_sched = schedules.from_config(spec_config.norm_voxel_lr_schedule)
  norm_voxel_ratio_sched = schedules.from_config(spec_config.norm_voxel_ratio_schedule)

  rng, key = random.split(rng)
  params = {}
  model, params['model'] = models.construct_nerf(
    key,
    use_hyper_spec_model=spec_config.use_hyper_spec_model,
    batch_size=train_config.batch_size,
    embeddings_dict=datasource.embeddings_dict,
    near=datasource.near,
    far=datasource.far,
    screw_input_mode=spec_config.screw_input_mode,
    use_sigma_gradient=spec_config.use_sigma_gradient,
    use_predicted_norm=spec_config.use_predicted_norm,
  )

  optimizer_def = optim.Adam(learning_rate_sched(0))
  if model.use_flow_model:
    focus = flax.traverse_util.ModelParamTraversal(lambda p, _: 'flow_model' not in p)
    optimizer = optimizer_def.create(params, focus=focus)
  else:
    optimizer = optimizer_def.create(params)

  # state = model_utils.TrainState(optimizer=optimizer)
  state = model_utils.TrainState(
    optimizer=optimizer,
    nerf_alpha=nerf_alpha_sched(0),
    warp_alpha=warp_alpha_sched(0),
    hyper_alpha=hyper_alpha_sched(0),
    hyper_sheet_alpha=hyper_sheet_alpha_sched(0),
    norm_loss_weight=norm_loss_weight_sched(0),
    norm_input_alpha=norm_input_alpha_sched(0),
    norm_voxel_lr=norm_voxel_lr_sched(0),
    norm_voxel_ratio=norm_voxel_ratio_sched(0),
  )
  scalar_params = training.ScalarParams(
    learning_rate=learning_rate_sched(0),
    elastic_loss_weight=elastic_loss_weight_sched(0),
    warp_reg_loss_weight=train_config.warp_reg_loss_weight,
    warp_reg_loss_alpha=train_config.warp_reg_loss_alpha,
    warp_reg_loss_scale=train_config.warp_reg_loss_scale,
    background_loss_weight=train_config.background_loss_weight,
    hyper_reg_loss_weight=train_config.hyper_reg_loss_weight,
    sigma_grad_diff_reg_weight=spec_config.sigma_grad_diff_reg_weight,
    back_facing_reg_weight=spec_config.back_facing_reg_weight,
    hyper_concentration_reg_weight=spec_config.hyper_concentration_reg_weight,
    hyper_concentration_reg_scale=spec_config.hyper_concentration_reg_scale,
    hyper_jacobian_reg_weight=spec_config.hyper_jacobian_reg_weight,
    hyper_jacobian_reg_scale=spec_config.hyper_jacobian_reg_scale,
    hyper_c_jacobian_reg_weight=spec_config.hyper_c_jacobian_reg_weight,
    hyper_c_jacobian_reg_scale=spec_config.hyper_c_jacobian_reg_scale,
    norm_voxel_loss_weight=spec_config.norm_voxel_loss_weight,
  )

  logging.info('Restoring checkpoint from %s', checkpoint_dir)
  state = checkpoints.restore_checkpoint(checkpoint_dir, state)
  step = state.optimizer.state.step + 1
  state = jax_utils.replicate(state, devices=devices_to_use)
  del params

  # @title Define pmapped render function.
  devices = jax.devices()

  def _model_fn(key_0, key_1, key_2, params, rays_dict, extra_params, norm_override):
    out = model.apply({'params': params},
                      rays_dict,
                      extra_params=extra_params,
                      rngs={
                        'coarse': key_0,
                        'fine': key_1,
                        'voxel': key_2
                      },
                      mutable=False,
                      screw_input_mode=spec_config.screw_input_mode,
                      use_sigma_gradient=spec_config.use_sigma_gradient,
                      use_predicted_norm=spec_config.use_predicted_norm,
                      return_points=False,
                      return_nv_details=False,
                      norm_voxel_ratio=1,  # inference ratio is always 1
                      mask_ratio=1,  # inference ratio is always 1
                      sharp_weights_std=0.1,
                      norm_override=norm_override
                      )
    return jax.lax.all_gather(out, axis_name='batch')

  pmodel_fn = jax.pmap(
      # Note rng_keys are useless in eval mode since there's no randomness.
      _model_fn,
      in_axes=(0, 0, 0, 0, 0, 0, None),  # Only distribute the data input.
      devices=devices_to_use,
      axis_name='batch',
  )
  render_fn = functools.partial(render_image,
                                model_fn=pmodel_fn,
                                device_count=len(devices),
                                chunk=chunk_size,
                                norm_override=norm_override)

  # @title Load cameras.
  camera_dir = Path(data_dir, camera_path_name)
  print(f'Loading cameras from {camera_dir}')
  test_camera_paths = datasource.glob_cameras(camera_dir)
  test_camera_paths = sort_camera_paths(test_camera_paths)
  test_cameras = utils.parallel_map(datasource.load_camera, test_camera_paths, show_pbar=True)

  mask_dir = Path(data_dir, 'resized_mask', f"{int(exp_config.image_scale)}x")
  print(f"Loading masks from {mask_dir}")
  mask_list = datasets.load_camera_masks(mask_dir, test_camera_paths, 1)  # already resized

  # @title Render video frames.
  rng = rng + jax.process_index()  # Make random seed separate across hosts.
  keys = random.split(rng, len(devices))

  results = []
  relevant_keys = ['rgb', 'med_depth', 'ray_norm', 'ray_delta_x', 'med_points', 'ray_predicted_mask']
  raw_result_list = []
  if interval == 1:
    camera_path_name += "_full"

  for i in tqdm(range(0, len(test_cameras), interval)):
  # for i in tqdm(range(0, 10, interval)):
    # print(f'Rendering frame {i + 1}/{len(test_cameras)}')
    camera = test_cameras[i]
    batch = datasets.camera_to_rays(camera)
    if not camera_path_name.startswith('vrig'):
      batch['metadata'] = {
        'appearance': jnp.ones_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) * i,
        'warp': jnp.ones_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) * i,
        'camera': jnp.ones_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) * 0
      }
    else:
      batch['metadata'] = {
        #         'appearance': jnp.ones_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) * ((i + 1) % 2 + 1),
        'appearance': jnp.ones_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) * i,
        'warp': jnp.ones_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) * i,
        'camera': jnp.ones_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) * ((i + 1) % 2 + 1)
      }
    mask = mask_list[i]
    batch['mask'] = mask

    render = render_fn(state, batch, rng=rng)

    # save raw results for future use
    raw_result = {}
    # value_size = 0
    for key in render:
      if not key in relevant_keys:
        continue
      raw_result[key] = np.array(render[key])
      # print(key, raw_result[key].size * raw_result[key].itemsize)
      # value_size += raw_result[key].size * raw_result[key].itemsize
    raw_result_list.append(raw_result)

    rgb = np.array(render['rgb'])
    depth_med = np.array(render['med_depth'])

    dummy_image = np.zeros_like(rgb)

    ray_norm = np.array(render['ray_norm'])
    ray_norm = model_utils.normalize_vector(ray_norm)
    ray_norm = ray_norm / 2.0 + 0.5

    ray_delta_x = np.array(render['ray_delta_x'])
    ray_delta_x = np.abs(ray_delta_x)
    ray_delta_x = ray_delta_x * 10

    med_points = np.array(render['med_points'])
    med_points = (med_points + 1.5) / 3     # -1.5 ~ 1.5 --> 0 ~ 1

    if 'ray_predicted_mask' in render:
      ray_predicted_mask = np.array(render['ray_predicted_mask'])
      ray_predicted_mask = np.broadcast_to(ray_predicted_mask, dummy_image.shape)  # grayscale to color
    else:
      ray_predicted_mask = dummy_image

    results.append((rgb, depth_med, ray_norm, ray_predicted_mask, ray_delta_x, med_points, dummy_image))

  # save raw render results
  raw_result_save_path = os.path.join(train_dir, "transfer_render_result_{}".format(camera_path_name))
  with open(raw_result_save_path, "wb+") as f:
    np.save(f, raw_result_list)

  # @title Show rendered video.
  fps = 30  # @param {type:'number'}
  rgb_frames = []
  debug_frames = []
  for rgb, depth_med, ray_norm, ray_predicted_mask, ray_delta_x, med_points, dummy_image in results:
    depth_viz = viz.colorize(depth_med.squeeze(), cmin=datasource.near, cmax=datasource.far, invert=True)
    med_points = med_points[..., :3].squeeze()

    row1 = np.concatenate([rgb, depth_viz, ray_norm], axis=1)
    row2 = np.concatenate([ray_predicted_mask, ray_delta_x, med_points], axis=1)
    debug_frame = np.concatenate([row1, row2], axis=0)
    debug_frames.append(image_utils.image_to_uint8(debug_frame))
    rgb_frames.append(image_utils.image_to_uint8(rgb))
  mediapy.set_show_save_dir(train_dir)
  mediapy.show_video(rgb_frames, fps=fps, title="transfer_result_{}_rgb".format(camera_path_name))
  mediapy.show_video(debug_frames, fps=fps, title="transfer_result_{}".format(camera_path_name))


def render_image(
    state,
    rays_dict,
    model_fn,
    device_count,
    rng,
    chunk=8192,
    default_ret_key=None,
    norm_override=None,
):
  """Render all the pixels of an image (in test mode).

  Args:
    state: model_utils.TrainState.
    rays_dict: dict, test example.
    model_fn: function, jit-ed render function.
    device_count: The number of devices to shard batches over.
    rng: The random number generator.
    chunk: int, the size of chunks to render sequentially.
    default_ret_key: either 'fine' or 'coarse'. If None will default to highest.

  Returns:
    rgb: jnp.ndarray, rendered color image.
    depth: jnp.ndarray, rendered depth.
    acc: jnp.ndarray, rendered accumulated weights per pixel.
  """
  batch_shape = rays_dict['origins'].shape[:-1]
  num_rays = np.prod(batch_shape)
  rays_dict = tree_util.tree_map(lambda x: x.reshape((num_rays, -1)), rays_dict)
  _, key_0, key_1, key_2 = jax.random.split(rng, 4)
  key_0 = jax.random.split(key_0, device_count)
  key_1 = jax.random.split(key_1, device_count)
  key_2 = jax.random.split(key_2, device_count)
  proc_id = jax.process_index()
  ret_maps = []
  start_time = time.time()
  num_batches = int(math.ceil(num_rays / chunk))
  logging.info('Rendering: num_batches = %d, num_rays = %d, chunk = %d',
               num_batches, num_rays, chunk)
  for batch_idx in range(num_batches):
    ray_idx = batch_idx * chunk
    logging.log_every_n_seconds(
        logging.INFO, 'Rendering batch %d/%d (%d/%d)', 2.0,
        batch_idx, num_batches, ray_idx, num_rays)
    # pylint: disable=cell-var-from-loop
    chunk_slice_fn = lambda x: x[ray_idx:ray_idx + chunk]
    chunk_rays_dict = tree_util.tree_map(chunk_slice_fn, rays_dict)
    num_chunk_rays = chunk_rays_dict['origins'].shape[0]
    remainder = num_chunk_rays % device_count
    if remainder != 0:
      padding = device_count - remainder
      # pylint: disable=cell-var-from-loop
      chunk_pad_fn = lambda x: jnp.pad(x, ((0, padding), (0, 0)), mode='edge')
      chunk_rays_dict = tree_util.tree_map(chunk_pad_fn, chunk_rays_dict)
    else:
      padding = 0
    # After padding the number of chunk_rays is always divisible by
    # proc_count.
    per_proc_rays = num_chunk_rays // jax.process_count()
    logging.debug(
        'Rendering batch: num_chunk_rays = %d, padding = %d, remainder = %d, '
        'per_proc_rays = %d',
        num_chunk_rays, padding, remainder, per_proc_rays)
    chunk_rays_dict = tree_util.tree_map(
        lambda x: x[(proc_id * per_proc_rays):((proc_id + 1) * per_proc_rays)],
        chunk_rays_dict)
    chunk_rays_dict = utils.shard(chunk_rays_dict, device_count)
    model_out = model_fn(key_0, key_1, key_2, state.optimizer.target['model'],
                         chunk_rays_dict, state.extra_params, norm_override)
    if not default_ret_key:
      ret_key = 'fine' if 'fine' in model_out else 'coarse'
    else:
      ret_key = default_ret_key
    ret_map = jax_utils.unreplicate(model_out[ret_key])
    # assert ret_map.keys() == 0, ret_map.keys()
    ret_map = jax.tree_map(lambda x: utils.unshard(x, padding), ret_map)
    ret_maps.append(ret_map)
  ret_map = jax.tree_multimap(lambda *x: jnp.concatenate(x, axis=0), *ret_maps)
  logging.info('Rendering took %.04s', time.time() - start_time)
  out = {}
  for key, value in ret_map.items():
    out_shape = (*batch_shape, *value.shape[1:])
    logging.debug('Reshaping %s of shape %s to %s',
                  key, str(value.shape), str(out_shape))
    out[key] = value.reshape(out_shape)

  return out

def sort_camera_paths(camera_paths):
  camera_names = [path.stem for path in camera_paths]
  id_path_pairs = []
  for i in range(len(camera_names)):
    camera_name = camera_names[i]
    camera_path = camera_paths[i]
    try:
      camera_id = camera_name.split('_')[1]
      int(camera_id)
    except:
      camera_id = camera_name.split('_')[0]
      int(camera_id)
    id_path_pairs.append((camera_id, camera_path))
  id_path_pairs = sorted(id_path_pairs)
  camera_paths = [path for id, path in id_path_pairs]
  return camera_paths

def normalize_vector(vector):
  eps = torch.Tensor([1e-6])
  # eps = 1e-5
  # eps = jnp.ones_like(vector[..., None, 0]) * eps
  vector = vector / torch.sqrt(torch.max(torch.sum(vector**2, dim=-1, keepdim=True), eps))
  return vector

def visualize_norm(norm_image, title):
  norm_image = normalize_vector(norm_image)

  norm_image = norm_image / 2.0 + 0.5
  norm_image = norm_image.numpy()
  norm_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2RGB)
  cv2.imshow(title, norm_image)
  cv2.waitKey(-1)

def load_video(video_path):
  cap = cv2.VideoCapture(str(video_path))
  frame_list = []
  while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
      break
    frame_list.append(frame)
  cap.release()
  return frame_list

exp_root = Path("/home/zwyan/3d_cv/repos/hypernerf_barf/experiments")
# dataset_name = "021_basin_01_novel_view"
# dataset_raw_name = "021_basin_01"
# # exp_name = Path("021_bs01_nv_ms_exp36")
# exp_name = Path("021_bs01_nv_ref_exp01")
dataset_name = "028_plate_03_novel_view"
dataset_raw_name = "028_plate_03"
# exp_name = Path("028_p03_nv_ms_exp40")
# exp_name = Path("028_p03_nv_ref_exp01")
exp_name = Path("028_p03_nv_ref_exp02")
exp_dir = exp_root / exp_name
render_result_path = exp_dir / "render_result_vrig_camera"

data_collect_root = Path('/home/zwyan/3d_cv/data/hypernerf/collect')
data_collect_dir = data_collect_root / dataset_raw_name

interval = 100
source_idx = 0    # index after the skipping
use_val = True
suffix = 'right' if use_val else 'left'

# # load mask
# mask_list = []
# mask_dir = data_collect_dir / 'resized_mask' / (suffix + "_part")
# for mask_path in sorted(glob.glob(str(mask_dir / '*.png'))):
#   mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)      # 0 is background, 255 is foreground
#   mask_list.append(mask)
#
# # skip mask
# selected_mask_list = []
# for i in range(0, len(mask_list), interval):
#   selected_mask_list.append(mask_list[i])

# # load render result
# render_result = np.load(render_result_path, allow_pickle=True)
# ray_norm_list = []
# for result in render_result:
#   ray_norm = result['ray_norm']                   # H x W x 3
#   ray_norm_list.append(ray_norm)
# ray_norm_override = ray_norm_list[source_idx]     # H x W x 3
#
# # mask the ray_norm
# # source_mask = selected_mask_list[source_idx]
# source_mask = np.zeros([ray_norm_list[0].shape[0], ray_norm_list[0].shape[1]])
# source_mask = torch.from_numpy(source_mask)
# source_mask = source_mask == 0                    # foreground is True, background is False
# source_mask = source_mask.float()                 # foreground is 1, background is 0
#
# ray_norm_override = torch.from_numpy(ray_norm_override)
# # visualize_norm(ray_norm_override, 'full')
# ray_norm_override = ray_norm_override * source_mask[..., None]
# # visualize_norm(ray_norm_override, 'masked')
#
# # take the masked average of the ray norm
# ray_norm_override = ray_norm_override.reshape([-1, 3])
# ray_norm_override = torch.sum(ray_norm_override, dim=0)
# source_mask_sum = torch.sum(source_mask)
# ray_norm_override = ray_norm_override / source_mask_sum       # 3
# # ray_norm_average = torch.ones_like(torch.Tensor(ray_norm_list[0])) * ray_norm_override
# # visualize_norm(ray_norm_average, 'average')
# ray_norm_override = jnp.array(ray_norm_override.numpy())      # torch to jax

ray_norm_override = jnp.array([1, 2, 3])

# render override norm
render_scene(dataset_name, exp_name, 'vrig_camera', interval, ray_norm_override)

# # load and compare the results
# rgb_video_name_original = 'result_vrig_camera_rgb.mp4'
# rgb_video_name_override = 'transfer_result_vrig_camera_rgb.mp4'
# rgb_video_path_original = exp_dir / rgb_video_name_original
# rgb_video_path_override = exp_dir / rgb_video_name_override
#
# rgb_list_original = load_video(rgb_video_path_original)
# rgb_list_override = load_video(rgb_video_path_override)
#
# # assert len(rgb_list_original) == len(rgb_list_override)
# for i in range(len(rgb_list_original)):
#   rgb_original = rgb_list_original[i] / 255.0
#   rgb_override = rgb_list_override[i] / 255.0
#   rgb_diff = np.abs(rgb_override - rgb_original)
#   rgb_concat = np.concatenate([rgb_override, rgb_original, rgb_diff], axis=0)
#   cv2.imshow('rgb_concat', rgb_concat)
#   key = cv2.waitKey(-1)
#   if key == 'q':
#     exit()

