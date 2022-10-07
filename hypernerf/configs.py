# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration classes."""
import dataclasses
from typing import Any, Callable, Optional

from flax import linen as nn
import gin
import immutabledict
import jax.numpy

from hypernerf import datasets

ScheduleDef = Any


gin.config.external_configurable(nn.elu, module='flax.nn')
gin.config.external_configurable(nn.relu, module='flax.nn')
gin.config.external_configurable(nn.leaky_relu, module='flax.nn')
gin.config.external_configurable(nn.tanh, module='flax.nn')
gin.config.external_configurable(nn.sigmoid, module='flax.nn')
gin.config.external_configurable(nn.softplus, module='flax.nn')
gin.config.external_configurable(nn.gelu, module='flax.nn')

gin.config.external_configurable(jax.numpy.sin, module='jax.numpy')
gin.config.external_configurable(jax.nn.relu, module='jax.nn')
gin.config.external_configurable(jax.nn.silu, module='jax.nn')
gin.config.external_configurable(jax.nn.gelu, module='jax.nn')



@gin.configurable()
@dataclasses.dataclass
class SpecularConfig:
  """Specular modelling configuration"""
  # whether screw axis is used in rgb rendering ["None", "rotation", "full"]
  screw_input_mode: str = None

  # whether the sigma gradient is used as surface norm
  use_sigma_gradient: bool = False

  # whether the sigma gradient diff from canonical and rotated observational is used for regularization
  use_sigma_grad_diff_reg: bool = False
  sigma_grad_diff_reg_weight: float = 0

  # whether to use the predicted surface norm as rgb condition
  use_predicted_norm: bool = False
  norm_loss_weight_schedule: ScheduleDef = immutabledict.immutabledict({
      'type': 'constant',
      'value': 0.001,
  })

  norm_input_alpha_schedule: ScheduleDef = immutabledict.immutabledict({
      'type': 'constant',
      'value': 4
  })

  use_back_facing_reg: bool = False
  back_facing_reg_weight: float = 0

  # w concentration loss
  use_hyper_concentration_reg_loss: bool = False
  hyper_concentration_reg_weight: float = 0
  hyper_concentration_reg_scale: float = 0  # scale c in Geman-McClure loss

  # hyper jacobian loss
  use_hyper_jacobian_reg_loss: bool = False
  hyper_jacobian_reg_weight: float = 0
  hyper_jacobian_reg_scale: float = 0  # scale c in Geman-McClure loss

  use_hyper_c_jacobian_reg_loss: bool = False
  hyper_c_jacobian_reg_weight: float = 0
  hyper_c_jacobian_reg_scale: float = 0  # scale c in Geman-McClure loss

  # norm voxel related
  norm_voxel_lr_schedule: ScheduleDef = immutabledict.immutabledict({
    'type': 'exponential',
    'initial_value': 0.1,
    'final_value': 1e-3,
    'num_steps': 250000,
  })
  norm_voxel_loss_weight: float = 0

  # the ratio between voxel norm and predicted norm
  # default use the norm voxel only
  norm_voxel_ratio_schedule: ScheduleDef = immutabledict.immutabledict({
    'type': 'constant',
    'value': 1
  })

  # hyper specular model
  use_hyper_spec_model: bool = False

  # mask related
  canonical_idx: int = 0

@gin.configurable()
@dataclasses.dataclass
class FlowConfig:
  """ Configurations for FlowModel """
  max_steps: int = 50000
  learning_rate_sched: ScheduleDef = immutabledict.immutabledict({
      'type': 'exponential',
      'initial_value': 1e-4,
      'final_value': 1e-6,
      'num_steps': max_steps,
  })
  # learning_rate_sched: ScheduleDef = immutabledict.immutabledict({
  #     'type': 'constant',
  #     'value': 1e-2,
  # })
  # learning_rate_sched: ScheduleDef = immutabledict.immutabledict({
  #     'type': 'constant',
  #     'value': 0,
  # })
  # time_override_sched: ScheduleDef = immutabledict.immutabledict({
  #   'type': 'linear',
  #   'initial_value': 1,
  #   'final_value': 1000,
  #   'num_steps': int(max_steps * 0.7)
  # })
  # time_override_sched: ScheduleDef = immutabledict.immutabledict({
  #   'type': 'constant',
  #   'value': 0
  # })
  time_override: int = 0
  warp_alpha_schedule: ScheduleDef = immutabledict.immutabledict({
    'type': 'linear',
    'initial_value': 0,
    'final_value': 4,
    'num_steps': 10000,
  })
  flow_model_light_learning_rate_sched: ScheduleDef = immutabledict.immutabledict({
      'type': 'constant',
      'value': 0,
  })
  elastic_loss_weight = 1

  print_every: int = 100
  log_every: int = 100
  save_every: int = 1000

@gin.configurable()
@dataclasses.dataclass
class ExperimentConfig:
  """Experiment configuration."""
  # A subname for the experiment e.g., for parameter sweeps. If this is set
  # experiment artifacts will be saves to a subdirectory with this name.
  subname: Optional[str] = None
  # The image scale to use for the dataset. Should be a power of 2.
  image_scale: int = 4
  # The random seed used to initialize the RNGs for the experiment.
  random_seed: int = 12345
  # The datasource class.
  datasource_cls: Callable[..., datasets.DataSource] = gin.REQUIRED


@gin.configurable()
@dataclasses.dataclass
class TrainConfig:
  """Parameters for training."""
  batch_size: int = gin.REQUIRED

  # The definition for the learning rate schedule.
  lr_schedule: ScheduleDef = immutabledict.immutabledict({
      'type': 'exponential',
      'initial_value': 0.001,
      'final_value': 0.0001,
      'num_steps': 1000000,
  })
  # The maximum number of training steps.
  max_steps: int = 1000000
  early_stop_steps: int = -1  # negative values for no early stop

  # Whether to use weight normalization.
  use_weight_norm: bool = False

  # The NeRF alpha schedule.
  nerf_alpha_schedule: Optional[ScheduleDef] = None
  # The warp alpha schedule.
  warp_alpha_schedule: Optional[ScheduleDef] = None
  # The schedule or the hyper sheet position encoding.
  hyper_alpha_schedule: Optional[ScheduleDef] = None
  # The schedule or the hyper sheet position encoding.
  hyper_sheet_alpha_schedule: Optional[ScheduleDef] = None

  # Whether to use the elastic regularization loss.
  use_elastic_loss: bool = False
  # The weight of the elastic regularization loss.
  elastic_loss_weight_schedule: Optional[ScheduleDef] = None
  # Which method to use to reduce the samples for the elastic loss.
  # 'weight' computes a weighted sum using the density weights, and 'median'
  # selects the sample at the median depth point.
  elastic_reduce_method: str = 'weight'
  # Which loss method to use for the elastic loss.
  elastic_loss_type: str = 'log_svals'
  # Whether to use background regularization.
  use_background_loss: bool = False
  # The weight for the background loss.
  background_loss_weight: float = 0.0
  # The batch size for background regularization loss.
  background_points_batch_size: int = 16384
  # Whether to use the warp reg loss.
  use_warp_reg_loss: bool = False
  # The weight for the warp reg loss.
  warp_reg_loss_weight: float = 0.0
  # The alpha for the warp reg loss.
  warp_reg_loss_alpha: float = -2.0
  # The scale for the warp reg loss.
  warp_reg_loss_scale: float = 0.001
  # Whether to regularize the hyper points.
  use_hyper_reg_loss: bool = False
  # The weight for the hyper reg loss.
  hyper_reg_loss_weight: float = 0.0

  # The size of the shuffle buffer size when shuffling the training dataset.
  # This needs to be sufficiently large to contain a diverse set of images in
  # each batch, especially when optimizing GLO embeddings.
  shuffle_buffer_size: int = 5000000
  # How often to save a checkpoint.
  save_every: int = 10000
  # How often to log to Tensorboard.
  log_every: int = 500
  # How often to log histograms to Tensorboard.
  histogram_every: int = 5000
  # How often to print to the console.
  print_every: int = 25

  # Unused, here for backwards compatibility.
  use_curvature_loss: bool = False
  curvature_loss_alpha: int = 0
  curvature_loss_scale: float = 0
  curvature_loss_spacing: float = 0
  curvature_loss_weight_schedule: Optional[Any] = None


@gin.configurable()
@dataclasses.dataclass
class EvalConfig:
  """Parameters for evaluation."""
  # If True only evaluate the model once, otherwise evaluate any new
  # checkpoints.
  eval_once: bool = False
  # If True save the predicted images to persistent storage.
  save_output: bool = True
  # The evaluation batch size.
  chunk: int = 8192
  # Max render checkpoints. The renders will rotate after this many.
  max_render_checkpoints: int = 3

  # The subname to append to 'renders' and 'summaries'.
  subname: str = ''

  # Unused args here for backwards compatibility.
  val_argmin: bool = False
  optim_metadata: bool = False
  optim_tile_size: int = 0
  optim_lr: float = 0.0

  # The number of validation examples to evaluate. (Default: all).
  num_val_eval: Optional[int] = 10
  # The number of training examples to evaluate.
  num_train_eval: Optional[int] = 10
  # The number of test examples to evaluate.
  num_test_eval: Optional[int] = 10
