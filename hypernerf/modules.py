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

"""Modules for NeRF models."""
import functools
from typing import Any, Optional, Tuple

import jax.lax as lax
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp

from hypernerf import model_utils
from hypernerf import types
from hypernerf.model_utils import get_trilinear_coefficient


def get_norm_layer(norm_type):
  """Translates a norm type to a norm constructor."""
  if norm_type is None or norm_type == 'none':
    return None
  elif norm_type == 'layer':
    return functools.partial(nn.LayerNorm, use_scale=False, use_bias=False)
  elif norm_type == 'group':
    return functools.partial(nn.GroupNorm, use_scale=False, use_bias=False)
  elif norm_type == 'batch':
    return functools.partial(nn.BatchNorm, use_scale=False, use_bias=False)
  else:
    raise ValueError(f'Unknown norm type {norm_type}')


class MLP(nn.Module):
  """Basic MLP class with hidden layers and an output layers."""
  depth: int
  width: int
  hidden_init: types.Initializer = jax.nn.initializers.glorot_uniform()
  hidden_activation: types.Activation = nn.relu
  hidden_norm: Optional[types.Normalizer] = None
  output_init: Optional[types.Initializer] = None
  output_channels: int = 0
  output_activation: Optional[types.Activation] = lambda x: x
  use_bias: bool = True
  skips: Tuple[int] = tuple()

  @nn.compact
  def __call__(self, x):
    inputs = x
    for i in range(self.depth):
      layer = nn.Dense(
        self.width,
        use_bias=self.use_bias,
        kernel_init=self.hidden_init,
        name=f'hidden_{i}')
      if i in self.skips:
        x = jnp.concatenate([x, inputs], axis=-1)
      x = layer(x)
      if self.hidden_norm is not None:
        x = self.hidden_norm()(x)  # pylint: disable=not-callable
      x = self.hidden_activation(x)

    if self.output_channels > 0:
      logit_layer = nn.Dense(
        self.output_channels,
        use_bias=self.use_bias,
        kernel_init=self.output_init,
        name='logit')
      x = logit_layer(x)
      if self.output_activation is not None:
        x = self.output_activation(x)

    return x


class NerfMLP(nn.Module):
  """A simple MLP.

  Attributes:
    nerf_trunk_depth: int, the depth of the first part of MLP.
    nerf_trunk_width: int, the width of the first part of MLP.
    nerf_rgb_branch_depth: int, the depth of the second part of MLP.
    nerf_rgb_branch_width: int, the width of the second part of MLP.
    activation: function, the activation function used in the MLP.
    skips: which layers to add skip layers to.
    alpha_channels: int, the number of alpha_channelss.
    rgb_channels: int, the number of rgb_channelss.
    condition_density: if True put the condition at the begining which
      conditions the density of the field.
  """
  trunk_depth: int = 8
  trunk_width: int = 256

  rgb_branch_depth: int = 1
  rgb_branch_width: int = 128
  rgb_channels: int = 3

  alpha_branch_depth: int = 0
  alpha_branch_width: int = 128
  alpha_channels: int = 1

  activation: types.Activation = nn.relu
  norm: Optional[Any] = None
  skips: Tuple[int] = (4,)

  predict_norm: bool = False
  predict_d_color: bool = False
  norm_dim: int = 3
  d_color_dim: int = 3
  s_weight_dim: int = 1

  def setup(self):
    dense = functools.partial(
      nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())
    self.bottleneck_layer = dense(self.trunk_width, name='bottleneck')

    self.trunk_mlp = MLP(depth=self.trunk_depth,
                         width=self.trunk_width,
                         hidden_activation=self.activation,
                         hidden_norm=self.norm,
                         hidden_init=jax.nn.initializers.glorot_uniform(),
                         skips=self.skips)
    self.rgb_mlp = MLP(depth=self.rgb_branch_depth,
                       width=self.rgb_branch_width,
                       hidden_activation=self.activation,
                       hidden_norm=self.norm,
                       hidden_init=jax.nn.initializers.glorot_uniform(),
                       output_init=jax.nn.initializers.glorot_uniform(),
                       output_channels=self.rgb_channels)
    # output of alpha mlp: sigma 1, norm 3, d_color 3, s_weight 1
    alpha_output_channel = self.alpha_channels
    if self.predict_norm:
      alpha_output_channel += self.norm_dim
    if self.predict_d_color:
      alpha_output_channel += self.d_color_dim + self.s_weight_dim
    self.alpha_mlp = MLP(depth=self.alpha_branch_depth,
                         width=self.alpha_branch_width,
                         hidden_activation=self.activation,
                         hidden_norm=self.norm,
                         hidden_init=jax.nn.initializers.glorot_uniform(),
                         output_init=jax.nn.initializers.glorot_uniform(),
                         output_channels=alpha_output_channel)

  def broadcast_condition(self, c, num_samples):
    # Broadcast condition from [batch, feature] to
    # [batch, num_coarse_samples, feature] since all the samples along the
    # same ray has the same viewdir.
    if len(c.shape) >= 2 and num_samples > 1:
      c = jnp.tile(c[:, None, :], (1, num_samples, 1))

    # Collapse the [batch, num_coarse_samples, feature] tensor to
    # [batch * num_coarse_samples, feature] to be fed into nn.Dense.
    c = c.reshape([-1, c.shape[-1]])
    return c

  @nn.compact
  def __call__(self, x, alpha_condition, rgb_condition, screw_condition=None):
    """Multi-layer perception for nerf.

    Args:
      x: sample points with shape [batch, num_coarse_samples, feature].
      alpha_condition: a condition array provided to the alpha branch.
      rgb_condition: a condition array provided in the RGB branch.

    Returns:
      raw: [batch, num_coarse_samples, rgb_channels+alpha_channels].
    """
    dense = functools.partial(
      nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())

    feature_dim = x.shape[-1]
    if len(x.shape) > 1:
      num_samples = x.shape[1]
    else:
      num_samples = 1
    x = x.reshape([-1, feature_dim])

    trunk_mlp = MLP(depth=self.trunk_depth,
                    width=self.trunk_width,
                    hidden_activation=self.activation,
                    hidden_norm=self.norm,
                    hidden_init=jax.nn.initializers.glorot_uniform(),
                    skips=self.skips)
    rgb_mlp = MLP(depth=self.rgb_branch_depth,
                  width=self.rgb_branch_width,
                  hidden_activation=self.activation,
                  hidden_norm=self.norm,
                  hidden_init=jax.nn.initializers.glorot_uniform(),
                  output_init=jax.nn.initializers.glorot_uniform(),
                  output_channels=self.rgb_channels)
    alpha_mlp = MLP(depth=self.alpha_branch_depth,
                    width=self.alpha_branch_width,
                    hidden_activation=self.activation,
                    hidden_norm=self.norm,
                    hidden_init=jax.nn.initializers.glorot_uniform(),
                    output_init=jax.nn.initializers.glorot_uniform(),
                    output_channels=self.alpha_channels)

    if self.trunk_depth > 0:
      x = trunk_mlp(x)

    if (alpha_condition is not None) or (rgb_condition is not None):
      bottleneck = dense(self.trunk_width, name='bottleneck')(x)

    if alpha_condition is not None:
      if alpha_condition.shape[0] != bottleneck.shape[0]:
        alpha_condition = self.broadcast_condition(alpha_condition, num_samples)
      alpha_input = jnp.concatenate([bottleneck, alpha_condition], axis=-1)
    else:
      alpha_input = x
    alpha = alpha_mlp(alpha_input)

    if rgb_condition is not None:
      if rgb_condition.shape[0] != bottleneck.shape[0]:
        rgb_condition = self.broadcast_condition(rgb_condition, num_samples)
      rgb_input = jnp.concatenate([bottleneck, rgb_condition], axis=-1)
    else:
      rgb_input = x

    if screw_condition is not None:
      screw_condition = jnp.reshape(screw_condition, [-1, screw_condition.shape[-1]])
      rgb_input = jnp.concatenate([rgb_input, screw_condition], axis=-1)
    else:
      rgb_input = rgb_input

    rgb = rgb_mlp(rgb_input)

    return {
      'rgb': rgb.reshape((-1, num_samples, self.rgb_channels)),
      'alpha': alpha.reshape((-1, num_samples, self.alpha_channels)),
    }

  def query_bottleneck(self, x, alpha_condition, rgb_condition):
    feature_dim = x.shape[-1]
    if len(x.shape) > 1:
      num_samples = x.shape[1]
    else:
      num_samples = 1
    x = x.reshape([-1, feature_dim])

    assert self.trunk_depth > 0
    trunk_output = self.trunk_mlp(x)

    if (alpha_condition is not None) or (rgb_condition is not None):
      bottleneck = self.bottleneck_layer(trunk_output)
    else:
      bottleneck = trunk_output
    return trunk_output, bottleneck

  def query_sigma(self, trunk_output, bottleneck, alpha_condition):
    feature_dim = trunk_output.shape[-1]
    if len(trunk_output.shape) > 1:
      num_samples = trunk_output.shape[1]
    else:
      num_samples = 1
    trunk_output = trunk_output.reshape([-1, feature_dim])

    if alpha_condition is not None:
      if alpha_condition.shape[0] != bottleneck.shape[0]:
        alpha_condition = self.broadcast_condition(alpha_condition, num_samples)
      alpha_input = jnp.concatenate([bottleneck, alpha_condition], axis=-1)
    else:
      alpha_input = trunk_output
    output = self.alpha_mlp(alpha_input)
    alpha = output[..., :self.alpha_channels]
    norm, d_color, s_weight = None, None, None
    if self.predict_norm:
      norm = output[..., self.alpha_channels: self.alpha_channels + self.norm_dim]
    if self.predict_d_color:
      d_color = output[...,
                self.alpha_channels + self.norm_dim:
                self.alpha_channels + self.norm_dim + self.d_color_dim]
      s_weight = output[...,
                 self.alpha_channels + self.norm_dim + self.d_color_dim:
                 self.alpha_channels + self.norm_dim + self.d_color_dim + self.s_weight_dim]
    return alpha, norm, d_color, s_weight

  def query_rgb(self, trunk_output, bottleneck, rgb_condition, screw_condition=None, norm=None, extra_rgb_condition=None):
    feature_dim = trunk_output.shape[-1]
    if len(trunk_output.shape) > 1:
      num_samples = trunk_output.shape[1]
    else:
      num_samples = 1
    trunk_output = trunk_output.reshape([-1, feature_dim])

    rgb_input = trunk_output
    if rgb_condition is not None:
      if rgb_condition.shape[0] != bottleneck.shape[0]:
        rgb_condition = self.broadcast_condition(rgb_condition, num_samples)
      rgb_input = jnp.concatenate([bottleneck, rgb_condition], axis=-1)
    if extra_rgb_condition is not None:
      rgb_input = jnp.concatenate([rgb_input, extra_rgb_condition], axis=-1)

    if screw_condition is not None:
      screw_condition = jnp.reshape(screw_condition, [-1, screw_condition.shape[-1]])
      rgb_input = jnp.concatenate([rgb_input, screw_condition], axis=-1)

    if norm is not None:
      norm = jnp.reshape(norm, [-1, norm.shape[-1]])
      rgb_input = jnp.concatenate([rgb_input, norm], axis=-1)

    rgb = self.rgb_mlp(rgb_input)
    return rgb


@gin.configurable(denylist=['name'])
class GLOEmbed(nn.Module):
  """A GLO encoder module, which is just a thin wrapper around nn.Embed.

  Attributes:
    num_embeddings: The number of embeddings.
    features: The dimensions of each embedding.
    embedding_init: The initializer to use for each.
  """

  num_embeddings: int = gin.REQUIRED
  num_dims: int = gin.REQUIRED
  embedding_init: types.Activation = nn.initializers.uniform(scale=0.05)

  def setup(self):
    self.embed = nn.Embed(
      num_embeddings=self.num_embeddings,
      features=self.num_dims,
      embedding_init=self.embedding_init)

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Method to get embeddings for specified indices.

    Args:
      inputs: The indices to fetch embeddings for.

    Returns:
      The embeddings corresponding to the indices provided.
    """
    if inputs.shape[-1] == 1:
      inputs = jnp.squeeze(inputs, axis=-1)

    return self.embed(inputs)


@gin.configurable(denylist=['name'])
class HyperSheetMLP(nn.Module):
  """An MLP that defines a bendy slicing surface through hyper space."""
  output_channels: int = gin.REQUIRED
  min_deg: int = 0
  max_deg: int = 1

  depth: int = 6
  width: int = 64
  skips: Tuple[int] = (4,)
  hidden_init: types.Initializer = jax.nn.initializers.glorot_uniform()
  output_init: types.Initializer = jax.nn.initializers.normal(1e-5)
  # output_init: types.Initializer = jax.nn.initializers.glorot_uniform()

  use_residual: bool = False

  @nn.compact
  def __call__(self, points, embed, alpha=None, use_embed=True, output_channel=None):
    """
    output_channel: override the attribute settings.
                    Only temporary implementation, until old experiments are no longer needed
    """
    points_feat = model_utils.posenc(
      points, self.min_deg, self.max_deg, alpha=alpha)
    if use_embed:
      inputs = jnp.concatenate([points_feat, embed], axis=-1)
    else:
      inputs = points_feat
    if output_channel is not None:
      output_channel_used = output_channel
    else:
      output_channel_used = self.output_channels
    mlp = MLP(depth=self.depth,
              width=self.width,
              skips=self.skips,
              hidden_init=self.hidden_init,
              output_channels=output_channel_used,
              output_init=self.output_init)
    if self.use_residual:
      return mlp(inputs) + embed
    else:
      return mlp(inputs)

@gin.configurable(denylist=['name'])
class MaskMLP(nn.Module):
  output_channels: int = 1
  min_deg: int = 0
  max_deg: int = 6

  depth: int = 6
  width: int = 64
  skips: Tuple[int] = (4,)
  hidden_init: types.Initializer = jax.nn.initializers.glorot_uniform()
  output_init: types.Initializer = jax.nn.initializers.normal(1e-5)
  # output_init: types.Initializer = jax.nn.initializers.glorot_uniform()

  output_activation: Optional[types.Activation] = lambda x: x

  @nn.compact
  def __call__(self, points, embed, alpha=None, use_embed=True, output_channel=None):
    """
    output_channel: override the attribute settings.
                    Only temporary implementation, until old experiments are no longer needed
    """
    points_feat = model_utils.posenc(
      points, self.min_deg, self.max_deg, alpha=alpha)
    if use_embed:
      inputs = jnp.concatenate([points_feat, embed], axis=-1)
    else:
      inputs = points_feat
    if output_channel is not None:
      output_channel_used = output_channel
    else:
      output_channel_used = self.output_channels
    mlp = MLP(depth=self.depth,
              width=self.width,
              skips=self.skips,
              hidden_init=self.hidden_init,
              output_channels=output_channel_used,
              output_init=self.output_init)
    outputs = mlp(inputs)
    if self.output_activation is not None:
      outputs = self.output_activation(outputs)
    return outputs

@gin.configurable(denylist=['name'])
class NormVoxels(nn.Module):
  voxel_shape: jnp.ndarray = gin.REQUIRED   # shape: (t, x, y, z, 3)
  range_x_min: jnp.float32 = -1.5
  range_x_max: jnp.float32 = 1.5
  range_y_min: jnp.float32 = -1.5
  range_y_max: jnp.float32 = 1.5
  range_z_min: jnp.float32 = -1.5
  range_z_max: jnp.float32 = 1.5

  def setup(self):
    # rng = self.make_rng('voxel')
    # self.voxel_array = self.param('norm_voxel_array', lambda rng, shape: jax.random.normal(rng, shape), self.voxel_shape)
    self.voxel_array = self.param('norm_voxel_array', lambda rng, shape: jnp.ones(shape) * jnp.sqrt(1/3.0), self.voxel_shape)
    # self.voxel_range = self.param('norm_voxel_range', lambda rng, shape: jnp.zeros(shape), (3, 2))

    # self.voxel_array = self.variable('params', 'norm_voxel_array', jax.random.normal, rng, self.voxel_shape)

    self.voxel_step_x = (self.range_x_max - self.range_x_min) / self.voxel_shape[1]
    self.voxel_step_y = (self.range_y_max - self.range_y_min) / self.voxel_shape[2]
    self.voxel_step_z = (self.range_z_max - self.range_z_min) / self.voxel_shape[3]

  def get_voxel_vertex_index(self, t, pos):
    """
    Input:
      t: shape N for N points
      pos: shape N x 3 for N points
    Output:
      index: shape N x 8 x 3, where each item is the (x,y,z) index for the vertex.
             vertex order is [C_000, C_100, C_010, C_110, C_001, C101, C011, C111]
    """
    # offset by the lower bound of voxels
    pos = pos - jnp.array([self.range_x_min, self.range_y_min, self.range_z_min])

    # get individual index, each of shape (N, )
    x_min_index = (pos[:, 0] // self.voxel_step_x).astype(jnp.int32)
    y_min_index = (pos[:, 1] // self.voxel_step_y).astype(jnp.int32)
    z_min_index = (pos[:, 2] // self.voxel_step_z).astype(jnp.int32)
    x_max_index = x_min_index + 1
    y_max_index = y_min_index + 1
    z_max_index = z_min_index + 1

    # cap at the boundary
    x_min_index = jnp.minimum(jnp.maximum(x_min_index, 0), self.voxel_shape[1] - 1)
    x_max_index = jnp.minimum(jnp.maximum(x_max_index, 0), self.voxel_shape[1] - 1)
    y_min_index = jnp.minimum(jnp.maximum(y_min_index, 0), self.voxel_shape[2] - 1)
    y_max_index = jnp.minimum(jnp.maximum(y_max_index, 0), self.voxel_shape[2] - 1)
    z_min_index = jnp.minimum(jnp.maximum(z_min_index, 0), self.voxel_shape[3] - 1)
    z_max_index = jnp.minimum(jnp.maximum(z_max_index, 0), self.voxel_shape[3] - 1)

    # assemble index for each vertex
    # each of shape (N, 3)
    c_000 = jnp.vstack([x_min_index, y_min_index, z_min_index]).transpose()
    c_100 = jnp.vstack([x_max_index, y_min_index, z_min_index]).transpose()
    c_010 = jnp.vstack([x_min_index, y_max_index, z_min_index]).transpose()
    c_110 = jnp.vstack([x_max_index, y_max_index, z_min_index]).transpose()
    c_001 = jnp.vstack([x_min_index, y_min_index, z_max_index]).transpose()
    c_101 = jnp.vstack([x_max_index, y_min_index, z_max_index]).transpose()
    c_011 = jnp.vstack([x_min_index, y_max_index, z_max_index]).transpose()
    c_111 = jnp.vstack([x_max_index, y_max_index, z_max_index]).transpose()

    # get all vertex spatial index
    # shape (N, 8, 3)
    vertex_index = jnp.concatenate([
      c_000[:, jnp.newaxis, :], c_100[:, jnp.newaxis, :], c_010[:, jnp.newaxis, :], c_110[:, jnp.newaxis, :],
      c_001[:, jnp.newaxis, :], c_101[:, jnp.newaxis, :], c_011[:, jnp.newaxis, :], c_111[:, jnp.newaxis, :]
    ], axis=1)

    # add t to vertex index
    t = jnp.tile(t[:, jnp.newaxis, jnp.newaxis], [1, 8, 1])
    vertex_index = jnp.concatenate([t, vertex_index], axis=-1)

    return vertex_index

  def get_vertex_values(self, vertex_index):
    """
    Input:
      vertex_index: N x 8 x 4     (N points, 8 vertex, (t, x, y, z))
    Output:
      vertex_value: N x 8 x 3
    """
    vertex_index = lax.stop_gradient(vertex_index.reshape([-1, 4]))    # 8N x 4
    t_index = vertex_index[:, 0]
    x_index = vertex_index[:, 1]
    y_index = vertex_index[:, 2]
    z_index = vertex_index[:, 3]

    values = self.voxel_array[tuple([t_index, x_index, y_index, z_index])]    # 8N x 3
    values = values.reshape([-1, 8, 3])
    return values

  # Jax array value update seems to take place out-place, resulting in the copying of the array
  def add_vertex_values(self, vertex_index, vertex_value):
    vertex_index = vertex_index.reshape([-1, 4])    # 8N x 4
    t_index = vertex_index[:, 0]
    x_index = vertex_index[:, 1]
    y_index = vertex_index[:, 2]
    z_index = vertex_index[:, 3]

    vertex_value = vertex_value.reshape([-1, 3])
    self.voxel_array.at[tuple([t_index, x_index, y_index, z_index])].add(vertex_value)

  def get_interpolation_coef(self, pos):
    """
    coef: value = sum(coef * [C_000, C_100, C_010, C_110, C_001, C101, C011, C111]^T)
          shape: N x 8
    """
    # offset by the lower bound of voxels
    pos = pos - jnp.array([self.range_x_min, self.range_y_min, self.range_z_min])

    # get normalized position
    step_array = jnp.array([self.voxel_step_x, self.voxel_step_y, self.voxel_step_z])
    pos_relative = jnp.mod(pos, step_array)
    pos_normalized = pos_relative / step_array

    interpolation_coef = get_trilinear_coefficient(pos_normalized)

    return interpolation_coef

  def get_interpolation_value(self, t, pos):
    coef = self.get_interpolation_coef(pos)
    vertex_index = self.get_voxel_vertex_index(t, pos)
    vertex_values = self.get_vertex_values(vertex_index)

    value = coef[:, :, jnp.newaxis] * vertex_values
    value = jnp.sum(value, axis=1)
    # value = jnp.mean(vertex_values, axis=1)
    return value, vertex_values, coef

  def get_and_update_value(self, t, pos, lr, sigma, target_norm):
    """
    get the interpolated value from the voxel array and then update the voxel array with the target norm
    Input:
      t: time, shape N
      pos: spatial position, shape N x 3
      lr: learning rate, shape 1
      sigma: occupancy, shape N
      target_norm: norm output from the mlp, shape N x 3
    Output:
      value: interpolated norm value, shape N x 3
    """
    coef = self.get_interpolation_coef(pos)
    vertex_index = self.get_voxel_vertex_index(t, pos)
    vertex_values = self.get_vertex_values(vertex_index)

    value = coef[:, :, jnp.newaxis] * vertex_values
    value = jnp.sum(value, axis=1)

    # update
    target_norm_expand = jnp.tile(jnp.expand_dims(target_norm, axis=1), [1, 8, 1])
    sigma_weight = (1 - jnp.exp(- sigma))[:, jnp.newaxis]
    distance_weight = coef
    update_lambda = jax.nn.sigmoid(lr * sigma_weight * distance_weight)[..., jnp.newaxis]

    new_vertex_values = update_lambda * target_norm_expand + (1 - update_lambda) * vertex_values
    diff_vertex_values = new_vertex_values - vertex_values
    self.add_vertex_values(vertex_index, diff_vertex_values)
    # add_vertex_values(self.voxel_array, vertex_index, diff_vertex_values)

    return value



