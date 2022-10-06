""" Train flow model from existing nerf """

import functools
from typing import Dict, Union, Callable, Any

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
from jax import numpy as jnp
from jax import random
import numpy as np
import tensorflow as tf

from hypernerf import configs, training_flow
from hypernerf import datasets
from hypernerf import gpath
from hypernerf import model_utils
from hypernerf import models
from hypernerf import schedules
from hypernerf import training
from hypernerf import utils
from hypernerf.training_flow import ScalarParams, train_step

flags.DEFINE_enum('mode', None, ['jax_cpu', 'jax_gpu', 'jax_tpu'],
                  'Distributed strategy approach.')

flags.DEFINE_string('base_folder', None, 'where to store ckpts and logs')
flags.mark_flag_as_required('base_folder')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
flags.DEFINE_multi_string('gin_configs', (), 'Gin config files.')
FLAGS = flags.FLAGS


def _log_to_tensorboard(writer: tensorboard.SummaryWriter,
                        state: model_utils.TrainState,
                        scalar_params: ScalarParams,
                        stats: Dict[str, Union[Dict[str, jnp.ndarray],
                                               jnp.ndarray]],
                        time_dict: Dict[str, jnp.ndarray]):
  """Log statistics to Tensorboard."""
  step = int(state.optimizer.state.step)

  def _log_scalar(tag, value):
    if value is not None:
      writer.scalar(tag, value, step)

  _log_scalar('params/learning_rate', scalar_params.learning_rate)
  _log_scalar('params/time_override', scalar_params.time_override)
  _log_scalar('params/warp_alpha', state.extra_params['warp_alpha'])
  _log_scalar('loss/flow/total', stats['loss/total'])
  _log_scalar('loss/flow/sigma', stats['loss/sigma'])
  _log_scalar('loss/flow/elastic', stats['loss/elastic'])

  for t, v in stats.items():
    if t.startswith('stats'):
      _log_scalar(t, v)

  for k, v in time_dict.items():
    writer.scalar(f'time/{k}', v, step)


def main(argv):
  jax.config.parse_flags_with_absl()
  tf.config.experimental.set_visible_devices([], 'GPU')
  del argv
  logging.info('*** Starting experiment')
  # Assume G3 path for config files when running locally.
  gin_configs = FLAGS.gin_configs

  logging.info('*** Loading Gin configs from: %s', str(gin_configs))
  gin.parse_config_files_and_bindings(
    config_files=gin_configs,
    bindings=FLAGS.gin_bindings,
    skip_unknown=True)

  # Load configurations.
  exp_config = configs.ExperimentConfig()
  train_config = configs.TrainConfig()
  spec_config = configs.SpecularConfig()
  flow_config = configs.FlowConfig()
  if spec_config.use_hyper_spec_model:
    dummy_model = models.HyperSpecModel({}, 0, 0)
  else:
    dummy_model = models.NerfModel({}, 0, 0)

  # Get directory information.
  exp_dir = gpath.GPath(FLAGS.base_folder)
  if exp_config.subname:
    exp_dir = exp_dir / exp_config.subname
  summary_dir = exp_dir / 'summaries' / 'train_flow'
  nerf_checkpoint_dir = exp_dir / 'checkpoints'
  flow_checkpoint_dir = exp_dir / 'checkpoints_flow'
  flow_only_checkpoint_dir = exp_dir / 'checkpoints_flow_only'

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
  scalar_params = ScalarParams(
    learning_rate=learning_rate_sched(0),
    time_override=flow_config.time_override,
    elastic_loss_weight=flow_config.elastic_loss_weight,
  )

  optimizer_def = optim.Adam(scalar_params.learning_rate)

  # load nerf model only
  nerf_params = {'model': nerf_params}
  nerf_optimizer = optimizer_def.create(nerf_params)
  nerf_state = model_utils.TrainState(optimizer=nerf_optimizer)

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
  init_step = int(flow_state.optimizer.state.step) + 1
  flow_state = jax_utils.replicate(flow_state, devices=devices)

  # Create Jax iterator.
  logging.info('Creating dataset iterator.')
  train_iter = datasource.create_iterator(
    datasource.train_ids,
    flatten=True,
    shuffle=True,
    batch_size=train_config.batch_size,
    prefetch_size=3,
    shuffle_buffer_size=train_config.shuffle_buffer_size,
    devices=devices,
  )
  points_iter = None

  summary_writer = None
  if jax.process_index() == 0:
    config_str = gin.operative_config_str()
    logging.info('Configuration: \n%s', config_str)
    with (exp_dir / 'config.gin').open('w') as f:
      f.write(config_str)
    summary_writer = tensorboard.SummaryWriter(str(summary_dir))
    summary_writer.text('gin/train', textdata=gin.markdown(config_str), step=0)

  train_step_fn = functools.partial(train_step, model)
  p_train_step_fn = jax.pmap(
    train_step_fn,
    axis_name='batch',
    devices=devices,
    # rng_key, state, batch, scalar_params, nerf_params
    in_axes=(0, 0, 0, None, 0),
    donate_argnums=(2, ),  # batch
  )

  if devices:
    n_local_devices = len(devices)
  else:
    n_local_devices = jax.local_device_count()

  logging.info('Starting training')
  # Make random seed separate across processes.
  rng = rng + jax.process_index()
  keys = random.split(rng, n_local_devices)
  time_tracker = utils.TimeTracker()
  time_tracker.tic('data', 'total')

  for step, batch in zip(range(init_step, flow_config.max_steps + 1),
                         train_iter):
    time_tracker.toc('data')
    # pytype: enable=attribute-error
    scalar_params = scalar_params.replace(
      learning_rate=learning_rate_sched(step),
    )
    warp_alpha = jax_utils.replicate(warp_alpha_sched(step), devices)
    flow_state = flow_state.replace(
                          warp_alpha=warp_alpha,
                          )

    with time_tracker.record_time('train_step'):
      flow_state, stats, keys, model_out = p_train_step_fn(
        keys, flow_state, batch, scalar_params, nerf_params)
      time_tracker.toc('total')

    if step % train_config.save_every == 0 and jax.process_index() == 0:
      training_flow.save_checkpoint(flow_checkpoint_dir, flow_only_checkpoint_dir, flow_state, keep=5)

    if step % flow_config.print_every == 0 and jax.process_index() == 0:
      logging.info('step=%d, loss=%0.6f, steps_per_second=%0.2f',
                   step, stats['loss/total'], time_tracker.summary()['steps_per_sec'])

    if step % train_config.log_every == 0 and jax.process_index() == 0:
      # Only log via process 0.
      _log_to_tensorboard(
          summary_writer,
          jax_utils.unreplicate(flow_state),
          scalar_params,
          jax_utils.unreplicate(stats),
          time_dict=time_tracker.summary('mean'))
      time_tracker.reset()
    time_tracker.tic('data', 'total')


if __name__ == '__main__':
  app.run(main)
