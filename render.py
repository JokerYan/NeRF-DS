import os
import sys

import jax
from jax.config import config as jax_config
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import flax
import flax.linen as nn
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints

import functools
from absl import logging
from io import BytesIO
import random as pyrandom
import numpy as np
import PIL
import IPython
import tempfile
import imageio
import mediapy
from IPython.display import display, HTML
from base64 import b64encode
from pathlib import Path
from pprint import pprint
import gin
from IPython.display import display, Markdown


from hypernerf import evaluation
from hypernerf import schedules
from hypernerf import training
from hypernerf import models
from hypernerf import modules
from hypernerf import warping
from hypernerf import configs
from hypernerf import datasets
from hypernerf import image_utils
from hypernerf import visualization as viz
from hypernerf import model_utils
from hypernerf import utils

######## parameter settings #########

dataset_name = 'aluminium-sheet-6/'
exp_name = 'as6_exp02'
camera_path_name = 'fix_camera_93'

#####################################


def render_scene(dataset_name, exp_name, camera_path_name,
                 start=0, end=float('inf'), interval=1):
  runtime_type = 'gpu'  # @param ['gpu', 'tpu']
  if runtime_type == 'tpu':
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()
  print('Detected Devices:', jax.devices())

  # @title Define imports and utility functions.
  # Monkey patch logging.
  def myprint(msg, *args, **kwargs):
    print(msg % args)

  logging.info = myprint
  logging.warn = myprint
  logging.error = myprint
  # @title Model and dataset configuration
  # @markdown Change the directories to where you saved your capture and experiment.
  project_root = './'
  if os.path.exists('/ssd/zhiwen/data/hypernerf/raw/'):
    data_root = '/ssd/zhiwen/data/hypernerf/raw/'
  elif os.path.exists('/hdd/zhiwen/data/hypernerf/raw/'):
    data_root = '/hdd/zhiwen/data/hypernerf/raw/'
  elif os.path.exists('/home/zwyan/3d_cv/data/hypernerf/raw/'):
    data_root = '/home/zwyan/3d_cv/data/hypernerf/raw/'
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
  mediapy.show_image(datasource.load_rgb(datasource.train_ids[0]))

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
  rng, key = random.split(rng)
  params = {}
  model, params['model'] = models.construct_nerf(
    key,
    batch_size=train_config.batch_size,
    embeddings_dict=datasource.embeddings_dict,
    near=datasource.near,
    far=datasource.far,
    screw_input_mode=spec_config.screw_input_mode,
    use_sigma_gradient=spec_config.use_sigma_gradient,
    use_predicted_norm=spec_config.use_predicted_norm,
  )
  optimizer_def = optim.Adam(learning_rate_sched(0))
  optimizer = optimizer_def.create(params)
  state = model_utils.TrainState(
    optimizer=optimizer,
    nerf_alpha=nerf_alpha_sched(0),
    warp_alpha=warp_alpha_sched(0),
    hyper_alpha=hyper_alpha_sched(0),
    hyper_sheet_alpha=hyper_sheet_alpha_sched(0),
    norm_loss_weight=norm_loss_weight_sched(0)
  )
  logging.info('Restoring checkpoint from %s', checkpoint_dir)
  state = checkpoints.restore_checkpoint(checkpoint_dir, state)
  step = state.optimizer.state.step + 1
  state = jax_utils.replicate(state, devices=devices_to_use)
  del params

  # @title Define pmapped render function.
  devices = jax.devices()
  def _model_fn(key_0, key_1, params, rays_dict, extra_params):
    out = model.apply({'params': params},
                      rays_dict,
                      extra_params=extra_params,
                      rngs={
                        'coarse': key_0,
                        'fine': key_1
                      },
                      mutable=False,
                      screw_input_mode=spec_config.screw_input_mode,
                      use_sigma_gradient=spec_config.use_sigma_gradient,
                      use_predicted_norm=spec_config.use_predicted_norm,
                      return_points=False
                      )
    return jax.lax.all_gather(out, axis_name='batch')

  pmodel_fn = jax.pmap(
    # Note rng_keys are useless in eval mode since there's no randomness.
    _model_fn,
    in_axes=(0, 0, 0, 0, 0),  # Only distribute the data input.
    devices=devices_to_use,
    axis_name='batch',
  )
  render_fn = functools.partial(evaluation.render_image,
                                model_fn=pmodel_fn,
                                device_count=len(devices),
                                chunk=eval_config.chunk)

  # @title Load cameras.
  camera_dir = Path(data_dir, camera_path_name)
  print(f'Loading cameras from {camera_dir}')
  test_camera_paths = datasource.glob_cameras(camera_dir)
  test_cameras = utils.parallel_map(datasource.load_camera, test_camera_paths, show_pbar=True)

  # @title Render video frames.
  rng = rng + jax.process_index()  # Make random seed separate across hosts.
  keys = random.split(rng, len(devices))
  results = []
  raw_result_list = []
  render_start = max(start, 0)
  render_end = int(min(end, len(test_cameras)))
  for i in range(render_start, render_end, interval):
    # for i in range(330, 400):
    print(f'Rendering frame {i + 1}/{len(test_cameras)}')
    camera = test_cameras[i]
    batch = datasets.camera_to_rays(camera)
    batch['metadata'] = {
      'appearance': jnp.ones_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) * i,
      'warp': jnp.ones_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) * i,
    }

    render = render_fn(state, batch, rng=rng)

    # save raw render results
    raw_result = {}
    for key in render:
      raw_result[key] = np.array(render[key])
    raw_result_list.append(raw_result)

    rgb = np.array(render['rgb'])
    depth_med = np.array(render['med_depth'])

    sigma_gradient = np.array(render['ray_sigma_gradient'])
    sigma_gradient = model_utils.normalize_vector(sigma_gradient)
    sigma_gradient = sigma_gradient / 2.0 + 0.5

    sigma_gradient_r = np.array(render['ray_sigma_gradient_r'])
    sigma_gradient_r = sigma_gradient_r / 2.0 + 0.5

    ray_rotation_field = np.array(render['ray_rotation_field'])
    ray_rotation_field = ray_rotation_field / 2.0 + 0.5

    ray_translation_field = np.array(render['ray_translation_field'])
    ray_translation_field = np.abs(ray_translation_field) / 0.03

    ray_hyper_points = np.array(render['ray_hyper_points'])
    ray_hyper_points = np.abs(ray_hyper_points)
    dummy_hyper = np.ones([ray_hyper_points.shape[0], ray_hyper_points.shape[1], 1]) * 0
    ray_hyper_points = np.concatenate([ray_hyper_points, dummy_hyper], axis=-1)

    ray_hyper_c = np.array(render['ray_hyper_c'])
    ray_hyper_c = np.abs(ray_hyper_c)
    dummy_hyper_c = np.ones([ray_hyper_c.shape[0], ray_hyper_c.shape[1], 1]) * 0
    ray_hyper_c = np.concatenate([ray_hyper_c, dummy_hyper_c], axis=-1)

    dummy_image = np.zeros_like(rgb)

    results.append((rgb, depth_med, sigma_gradient, sigma_gradient_r,
                    ray_rotation_field, ray_translation_field, ray_hyper_points, ray_hyper_c, dummy_image))
    depth_viz = viz.colorize(depth_med.squeeze(), cmin=datasource.near, cmax=datasource.far, invert=True)
    # mediapy.show_images([rgb, depth_viz, sigma_gradient, sigma_gradient_r,
    #                      ray_rotation_field, ray_translation_field, ray_hyper_points, ray_hyper_c], columns=4)

  # @title Show rendered video.
  fps = 30  # @param {type:'number'}
  frames = []
  for rgb, depth, sigma_gradient, sigma_gradient_r, ray_rotation_field, ray_translation_field, ray_hyper_points, ray_hyper_c, dummy_image in results:
    depth_viz = viz.colorize(depth.squeeze(), cmin=datasource.near, cmax=datasource.far, invert=True)
    #   frame = np.concatenate([rgb, depth_viz, sigma_gradient, sigma_gradient_r,
    #                           ray_rotation_field, ray_hyper_points, ray_hyper_c], axis=1)
    row1 = np.concatenate([rgb, depth_viz, sigma_gradient, sigma_gradient_r], axis=1)
    row2 = np.concatenate([ray_rotation_field, ray_translation_field, ray_hyper_points, ray_hyper_c], axis=1)
    frame = np.concatenate([row1, row2], axis=0)
    frames.append(image_utils.image_to_uint8(frame))
  mediapy.set_show_save_dir(train_dir)
  mediapy.show_video(frames, fps=fps, title="result_{}".format(camera_path_name))

  # save raw render results
  raw_result_save_path = os.path.join(train_dir, "render_result_{}".format(camera_path_name))
  with open(raw_result_save_path, "wb+") as f:
    np.save(f, raw_result_list)


if __name__ == "__main__":
  render_scene(dataset_name, exp_name, camera_path_name)



