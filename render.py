import os

import jax
import jax.numpy as jnp
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
import gin
from IPython.display import display, Markdown
from tqdm import tqdm


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

######## parameter settings #########

dataset_name = '015_cup_02_novel_view'
exp_name = '015_c02_nv_ms_exp36'
camera_path_name = 'vrig_camera'
interval = 100
chunk_size = 2048

#####################################


def render_scene(dataset_name, exp_name, camera_path_name, interval):
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

  # param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
  # print("Total number of params:", param_count)
  del params

  # @title Define pmapped render function.
  devices = jax.devices()

  def _model_fn(key_0, key_1, key_2, params, rays_dict, extra_params):
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
                      sharp_weights_std=0.1
                      )
    return jax.lax.all_gather(out, axis_name='batch')

  pmodel_fn = jax.pmap(
      # Note rng_keys are useless in eval mode since there's no randomness.
      _model_fn,
      in_axes=(0, 0, 0, 0, 0, 0),  # Only distribute the data input.
      devices=devices_to_use,
      axis_name='batch',
  )
  # pmodel_fn = jax.vmap(
  #     # Note rng_keys are useless in eval mode since there's no randomness.
  #     _model_fn,
  #     in_axes=(0, 0, 0, 0, 0, 0),  # Only distribute the data input.
  #     # devices=devices_to_use,
  #     axis_name='batch',
  # )
  render_fn = functools.partial(evaluation.render_image,
                                model_fn=pmodel_fn,
                                device_count=len(devices),
                                chunk=chunk_size)

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
  relevant_keys = ['rgb', 'med_depth', 'ray_norm', 'ray_delta_x', 'med_points',
                   'ray_predicted_mask', 'ray_rotation_field']
  raw_result_list = []
  if interval == 1:
    camera_path_name += "_full"

  for i in tqdm(range(0, len(test_cameras), interval)):
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
  #
  #   # save raw results for future use
  #   raw_result = {}
  #   # value_size = 0
  #   for key in render:
  #     if not key in relevant_keys:
  #       continue
  #     raw_result[key] = np.array(render[key])
  #     # print(key, raw_result[key].size * raw_result[key].itemsize)
  #     # value_size += raw_result[key].size * raw_result[key].itemsize
  #   raw_result_list.append(raw_result)
  #
  #   rgb = np.array(render['rgb'])
  #   depth_med = np.array(render['med_depth'])
  #
  #   dummy_image = np.zeros_like(rgb)
  #
  #   ray_norm = np.array(render['ray_norm'])
  #   ray_norm = model_utils.normalize_vector(ray_norm)
  #   ray_norm = ray_norm / 2.0 + 0.5
  #
  #   ray_delta_x = np.array(render['ray_delta_x'])
  #   ray_delta_x = np.abs(ray_delta_x)
  #   ray_delta_x = ray_delta_x * 10
  #
  #   med_points = np.array(render['med_points'])
  #   med_points = (med_points + 1.5) / 3     # -1.5 ~ 1.5 --> 0 ~ 1
  #
  #   if 'ray_predicted_mask' in render:
  #     ray_predicted_mask = np.array(render['ray_predicted_mask'])
  #     ray_predicted_mask = np.broadcast_to(ray_predicted_mask, dummy_image.shape)  # grayscale to color
  #   else:
  #     ray_predicted_mask = dummy_image
  #
  #   results.append((rgb, depth_med, ray_norm, ray_predicted_mask, ray_delta_x, med_points, dummy_image))
  #
  # # save raw render results
  # raw_result_save_path = os.path.join(train_dir, "render_result_{}".format(camera_path_name))
  # with open(raw_result_save_path, "wb+") as f:
  #   np.save(f, raw_result_list)
  #
  # # @title Show rendered video.
  # fps = 30  # @param {type:'number'}
  # rgb_frames = []
  # debug_frames = []
  # for rgb, depth_med, ray_norm, ray_predicted_mask, ray_delta_x, med_points, dummy_image in results:
  #   depth_viz = viz.colorize(depth_med.squeeze(), cmin=datasource.near, cmax=datasource.far, invert=True)
  #   med_points = med_points[..., :3].squeeze()
  #
  #   row1 = np.concatenate([rgb, depth_viz, ray_norm], axis=1)
  #   row2 = np.concatenate([ray_predicted_mask, ray_delta_x, med_points], axis=1)
  #   debug_frame = np.concatenate([row1, row2], axis=0)
  #   debug_frames.append(image_utils.image_to_uint8(debug_frame))
  #   rgb_frames.append(image_utils.image_to_uint8(rgb))
  # mediapy.set_show_save_dir(train_dir)
  # mediapy.show_video(rgb_frames, fps=fps, title="result_{}_rgb".format(camera_path_name))
  # mediapy.show_video(debug_frames, fps=fps, title="result_{}".format(camera_path_name))


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


if __name__ == "__main__":
  render_scene(dataset_name, exp_name, camera_path_name, interval)



