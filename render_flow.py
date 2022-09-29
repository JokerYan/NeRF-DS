""" Train flow model from existing nerf """

import os
import functools
from typing import Dict, Union, Callable, Any

import cv2
import mediapy
from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import optim
from flax import struct
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
import jax
from jax import numpy as jnp, lax
from jax import random
import numpy as np
import tensorflow as tf
from pathlib import Path

from hypernerf import configs, evaluation
from hypernerf import datasets
from hypernerf import gpath
from hypernerf import model_utils
from hypernerf import models
from hypernerf import schedules
from hypernerf import training
from hypernerf import utils
from hypernerf.training_flow import ScalarParams, train_step


######## parameter settings #########

dataset_name = 'bell-2_qualitative/'
exp_name = 'b2_q_ref_exp01'
camera_path_name = 'fix_camera_93'

start = 0
end = float('inf')
interval = 500

#####################################


def render_scene(argv):
  del argv

  project_root = './'
  if os.path.exists('/ssd/zhiwen/data/hypernerf/raw/'):
    data_root = '/ssd/zhiwen/data/hypernerf/raw/'
  elif os.path.exists('/hdd/zhiwen/data/hypernerf/raw/'):
    data_root = '/hdd/zhiwen/data/hypernerf/raw/'
  elif os.path.exists('/home/zwyan/3d_cv/data/hypernerf/raw/'):
    data_root = '/home/zwyan/3d_cv/data/hypernerf/raw/'
  else:
    raise NotImplemented

  # Load configurations.
  train_dir = os.path.join(project_root, 'experiments', exp_name)
  config_path = Path(train_dir, 'config.gin')
  with open(config_path, 'r') as f:
    logging.info('Loading config from %s', config_path)
    config_str = f.read()
  gin.parse_config(config_str)

  exp_config = configs.ExperimentConfig()
  train_config = configs.TrainConfig()
  spec_config = configs.SpecularConfig()
  flow_config = configs.FlowConfig()
  if spec_config.use_hyper_spec_model:
    dummy_model = models.HyperSpecModel({}, 0, 0)
  else:
    dummy_model = models.NerfModel({}, 0, 0)

  # Get directory information.
  exp_dir = gpath.GPath(train_dir)
  if exp_config.subname:
    exp_dir = exp_dir / exp_config.subname
  summary_dir = exp_dir / 'summaries' / 'train_flow'
  nerf_checkpoint_dir = exp_dir / 'checkpoints'
  flow_checkpoint_dir = exp_dir / 'checkpoints_flow'

  # Log and create directories if this is the main process.
  if jax.process_index() == 0:
    logging.info('exp_dir = %s', exp_dir)
    if not exp_dir.exists():
      exp_dir.mkdir(parents=True, exist_ok=True)

    logging.info('summary_dir = %s', summary_dir)
    if not summary_dir.exists():
      summary_dir.mkdir(parents=True, exist_ok=True)

    logging.info('nerf_checkpoint_dir = %s', nerf_checkpoint_dir)
    if not nerf_checkpoint_dir.exists():
      nerf_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logging.info('flow_checkpoint_dir = %s', nerf_checkpoint_dir)
    if not flow_checkpoint_dir.exists():
      flow_checkpoint_dir.mkdir(parents=True, exist_ok=True)

  logging.info('Starting process %d. There are %d processes.',
               jax.process_index(), jax.process_count())
  logging.info('Found %d accelerator devices: %s.', jax.local_device_count(),
               str(jax.local_devices()))
  logging.info('Found %d total devices: %s.', jax.device_count(),
               str(jax.devices()))

  rng = random.PRNGKey(exp_config.random_seed)
  # Shift the numpy random seed by process_index() to shuffle data loaded by
  # different processes.
  np.random.seed(exp_config.random_seed + jax.process_index())

  if train_config.batch_size % jax.device_count() != 0:
    raise ValueError('Batch size must be divisible by the number of devices.')

  devices = jax.local_devices()
  logging.info('Creating datasource')
  datasource = exp_config.datasource_cls(
    image_scale=exp_config.image_scale,
    random_seed=exp_config.random_seed,
    # Enable metadata based on model needs.
    use_warp_id=dummy_model.use_warp,
    use_appearance_id=(
            dummy_model.nerf_embed_key == 'appearance'
            or dummy_model.hyper_embed_key == 'appearance'),
    use_camera_id=dummy_model.nerf_embed_key == 'camera',
    use_time=dummy_model.warp_embed_key == 'time')

  # Create Model.
  logging.info('Initializing models.')
  rng, key = random.split(rng)
  flow_model, nerf_params, flow_params = models.construct_flow(
    key,
    use_hyper_spec_model=spec_config.use_hyper_spec_model,
    batch_size=train_config.batch_size,
    embeddings_dict=datasource.embeddings_dict,
    near=datasource.near,
    far=datasource.far,
    screw_input_mode=spec_config.screw_input_mode,
    use_sigma_gradient=spec_config.use_sigma_gradient,
    use_predicted_norm=spec_config.use_predicted_norm
  )
  model = flow_model

  learning_rate_sched = schedules.from_config(flow_config.learning_rate_sched)
  warp_alpha_sched = schedules.from_config(flow_config.warp_alpha_schedule)
  time_offset_sched = schedules.from_config(flow_config.time_offset_sched)
  scalar_params = ScalarParams(
    learning_rate=learning_rate_sched(0),
    time_offset=time_offset_sched(0),
    elastic_loss_weight=flow_config.elastic_loss_weight,
  )

  optimizer_def = optim.Adam(scalar_params.learning_rate)

  # load nerf model only
  nerf_params = {'model': nerf_params}
  nerf_optimizer = optimizer_def.create(nerf_params)
  nerf_state = model_utils.TrainState(
    optimizer=nerf_optimizer,
    norm_voxel_lr=0,
    norm_voxel_ratio=0
  )

  logging.info('Restoring nerf checkpoint from %s', nerf_checkpoint_dir)
  nerf_state = checkpoints.restore_checkpoint(nerf_checkpoint_dir, nerf_state)
  nerf_state = jax_utils.replicate(nerf_state, devices=devices)
  nerf_params = nerf_state.optimizer.target

  # load flow model
  flow_params = {'model': flow_params}
  flow_optimizer = optimizer_def.create(flow_params)
  flow_state = model_utils.TrainState(
    optimizer=flow_optimizer,
    warp_alpha=warp_alpha_sched(0),
  )

  logging.info('Restoring flow checkpoint from %s', flow_checkpoint_dir)
  flow_state = checkpoints.restore_checkpoint(flow_checkpoint_dir, flow_state)
  flow_state = jax_utils.replicate(flow_state, devices=devices)

  # combine params
  params = {}
  params['model'] = flow_state.optimizer.target['model']
  params['model']['nerf_model'] = nerf_params['model']
  combined_optimizer = optimizer_def.create(params)
  combined_state = nerf_state
  combined_state = combined_state.replace(
    optimizer=combined_optimizer
  )

  if devices:
    n_local_devices = len(devices)
  else:
    n_local_devices = jax.local_device_count()

  def _model_fn(key_0, key_1, key_2, params, batch, extra_params):
    out = model.apply(
      {'params': params},
      batch,
      extra_params=extra_params,
      time_offset=0,
      rngs={
        'coarse': random.PRNGKey(0),
        'fine': random.PRNGKey(0),
      },
    )
    return jax.lax.all_gather(out, axis_name='batch')

  pmodel_fn = jax.pmap(
    # Note rng_keys are useless in eval mode since there's no randomness.
    _model_fn,
    in_axes=(0, 0, 0, 0, 0, 0),  # Only distribute the data input.
    devices=devices,
    axis_name='batch',
  )
  render_fn = functools.partial(evaluation.render_flow,
                                model_fn=pmodel_fn,
                                device_count=len(devices),
                                chunk=8192)

  # load cameras
  data_dir = os.path.join(data_root, dataset_name)  # @param {type: "string"}
  camera_dir = Path(data_dir, camera_path_name)
  print(f'Loading cameras from {camera_dir}')
  test_camera_paths = datasource.glob_cameras(camera_dir)
  test_cameras = utils.parallel_map(datasource.load_camera, test_camera_paths, show_pbar=True)

  # render
  rng = rng + jax.process_index()  # Make random seed separate across hosts.
  render_start = max(start, 0)
  render_end = int(min(end, len(test_cameras)))
  frames = []
  for i in range(render_start, render_end, interval):
    print(f'Rendering frame {i + 1}/{len(test_cameras)}')
    camera = test_cameras[i]
    batch = datasets.camera_to_rays(camera)
    batch['metadata'] = {
        'appearance': jnp.ones_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) * i,
        'warp': jnp.ones_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) * i,
        'camera': jnp.ones_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) * ((i + 1) % 2 + 1)
    }

    render = render_fn(combined_state, batch, rng=rng)

    weights = lax.stop_gradient(render['weights'])

    ray_delta_x = render['ray_delta_x']
    ray_delta_x = np.array(ray_delta_x)
    print(np.min(ray_delta_x), np.max(ray_delta_x), np.max(weights))
    ray_delta_x = np.abs(ray_delta_x)
    # ray_delta_x = (ray_delta_x - np.min(ray_delta_x)) / (np.max(ray_delta_x) - np.min(ray_delta_x))

    cur_sigma = render['cur_sigma']
    warped_sigma = render['warped_sigma']
    loss = (weights * jnp.abs(cur_sigma - warped_sigma)).sum(-1).mean()
    print(loss)

    frames.append(np.array(ray_delta_x * 255, np.uint8))
    # cv2.imshow("ray_delta_x", ray_delta_x)
    # cv2.waitKey()

  # save as video
  _fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  height, width = frames[0].shape[:2]
  video_writer = cv2.VideoWriter(os.path.join(train_dir, "flow.mp4"), _fourcc, 30.0, (width, height))
  for frame in frames:
    video_writer.write(frame)
  video_writer.release()
  print("Flow video saved to {}".format(os.path.join(train_dir, "flow.mp4")))


if __name__ == '__main__':
  app.run(render_scene)
