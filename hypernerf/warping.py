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

"""Warp fields."""
from typing import Any, Iterable, Optional, Dict

from flax import linen as nn
import gin
import jax
import jax.numpy as jnp

from hypernerf import model_utils
from hypernerf import utils
from hypernerf import modules
from hypernerf import rigid_body as rigid
from hypernerf import types
from hypernerf.bone_utils import get_bone_probs_batch
from hypernerf.quaternion import to_rotation_matrix


@gin.configurable(denylist=['name'])
class TranslationField(nn.Module):
  """Network that predicts warps as a translation field.

  References:
    https://en.wikipedia.org/wiki/Vector_potential
    https://en.wikipedia.org/wiki/Helmholtz_decomposition

  Attributes:
    metadata_encoder: an encoder for metadata.
    alpha: the alpha for the positional encoding.
    skips: the index of the layers with skip connections.
    depth: the depth of the network excluding the output layer.
    hidden_channels: the width of the network hidden layers.
    activation: the activation for each layer.
    metadata_encoded: whether the metadata parameter is pre-encoded or not.
    hidden_initializer: the initializer for the hidden layers.
    output_initializer: the initializer for the last output layer.
  """
  min_deg: int = 0
  max_deg: int = 8
  use_posenc_identity: bool = True

  skips: Iterable[int] = (4,)
  depth: int = 6
  hidden_channels: int = 128
  activation: types.Activation = nn.relu
  norm: Optional[Any] = None
  hidden_init: types.Initializer = jax.nn.initializers.glorot_uniform()
  output_init: types.Initializer = jax.nn.initializers.uniform(scale=1e-4)

  def setup(self):
    # Note that this must be done this way instead of using mutable list
    # operations.
    # See https://github.com/google/flax/issues/524.
    # pylint: disable=g-complex-comprehension
    output_dims = 3
    self.mlp = modules.MLP(
        width=self.hidden_channels,
        depth=self.depth,
        skips=self.skips,
        hidden_activation=self.activation,
        hidden_norm=self.norm,
        hidden_init=self.hidden_init,
        output_init=self.output_init,
        output_channels=output_dims)

  def warp(self,
           points: jnp.ndarray,
           metadata: jnp.ndarray,
           extra_params: Dict[str, Any]):
    points_embed = model_utils.posenc(points,
                                      min_deg=self.min_deg,
                                      max_deg=self.max_deg,
                                      use_identity=self.use_posenc_identity,
                                      alpha=extra_params['warp_alpha'])
    inputs = jnp.concatenate([points_embed, metadata], axis=-1)
    translation = self.mlp(inputs)
    warped_points = points + translation

    return warped_points

  def __call__(self,
               points: jnp.ndarray,
               metadata: jnp.ndarray,
               extra_params: Dict[str, Any],
               return_jacobian: bool = False):
    """Warp the given points using a warp field.

    Args:
      points: the points to warp.
      metadata: encoded metadata features.
      extra_params: extra parameters used in the warp field e.g., the warp
        alpha.
      return_jacobian: if True compute and return the Jacobian of the warp.

    Returns:
      The warped points and the Jacobian of the warp if `return_jacobian` is
        True.
    """
    out = {
        'warped_points': self.warp(points, metadata, extra_params)
    }

    if return_jacobian:
      jac_fn = jax.jacfwd(lambda *x: self.warp(*x)[..., :3], argnums=0)
      out['jacobian'] = jac_fn(points, metadata, extra_params)

    return out


@gin.configurable(denylist=['name'])
class SE3Field(nn.Module):
  """Network that predicts warps as an SE(3) field.

  Attributes:
    points_encoder: the positional encoder for the points.
    metadata_encoder: an encoder for metadata.
    alpha: the alpha for the positional encoding.
    skips: the index of the layers with skip connections.
    depth: the depth of the network excluding the logit layer.
    hidden_channels: the width of the network hidden layers.
    activation: the activation for each layer.
    metadata_encoded: whether the metadata parameter is pre-encoded or not.
    hidden_initializer: the initializer for the hidden layers.
    output_initializer: the initializer for the last logit layer.
  """
  min_deg: int = 0
  max_deg: int = 8
  use_posenc_identity: bool = False

  activation: types.Activation = nn.relu
  norm: Optional[Any] = None
  skips: Iterable[int] = (4,)
  trunk_depth: int = 6
  trunk_width: int = 128
  rotation_depth: int = 0
  rotation_width: int = 128
  pivot_depth: int = 0
  pivot_width: int = 128
  translation_depth: int = 0
  translation_width: int = 128

  default_init: types.Initializer = jax.nn.initializers.xavier_uniform()
  rotation_init: types.Initializer = jax.nn.initializers.uniform(scale=1e-4)
  translation_init: types.Initializer = jax.nn.initializers.uniform(scale=1e-4)

  # Unused, here for backwards compatibility.
  num_hyper_dims: int = 0
  hyper_depth: int = 0
  hyper_width: int = 0
  hyper_init: Optional[types.Initializer] = None

  def setup(self):
    self.trunk = modules.MLP(
        depth=self.trunk_depth,
        width=self.trunk_width,
        hidden_activation=self.activation,
        hidden_norm=self.norm,
        hidden_init=self.default_init,
        skips=self.skips)

    branches = {
        'w':
            modules.MLP(
                depth=self.rotation_depth,
                width=self.rotation_width,
                hidden_activation=self.activation,
                hidden_norm=self.norm,
                hidden_init=self.default_init,
                output_init=self.rotation_init,
                output_channels=3),
        'v':
            modules.MLP(
                depth=self.translation_depth,
                width=self.translation_width,
                hidden_activation=self.activation,
                hidden_norm=self.norm,
                hidden_init=self.default_init,
                output_init=self.translation_init,
                output_channels=3),
    }

    # Note that this must be done this way instead of using mutable operations.
    # See https://github.com/google/flax/issues/524.
    self.branches = branches

  # if vector is none, transform points, else transform vector
  def warp(self,
           points: jnp.ndarray,
           metadata_embed: jnp.ndarray,
           extra_params: Dict[str, Any],
           return_screw: bool = False,
           vector: jnp.ndarray = None,
           inverse: bool = False,
           with_translation: bool = False
           ):
    points_embed = model_utils.posenc(points,
                                      min_deg=self.min_deg,
                                      max_deg=self.max_deg,
                                      use_identity=self.use_posenc_identity,
                                      alpha=extra_params['warp_alpha'])
    inputs = jnp.concatenate([points_embed, metadata_embed], axis=-1)
    trunk_output = self.trunk(inputs)

    w = self.branches['w'](trunk_output)
    v = self.branches['v'](trunk_output)
    theta = jnp.linalg.norm(w, axis=-1)
    w = w / theta[..., jnp.newaxis]
    v = v / theta[..., jnp.newaxis]
    screw_axis = jnp.concatenate([w, v], axis=-1)

    rotation_only = vector is not None and not with_translation
    transform = rigid.exp_se3(screw_axis, theta, rotation_only=rotation_only, inverse=inverse)

    if vector is None:
      warped_points = points
    else:
      warped_points = vector
    warped_points = rigid.from_homogenous(
        utils.matmul(transform, rigid.to_homogenous(warped_points)))

    if return_screw:
      return warped_points, screw_axis
    else:
      return warped_points, None

  def __call__(self,
               points: jnp.ndarray,
               metadata: jnp.ndarray,
               extra_params: Dict[str, Any],
               return_jacobian: bool = False,
               vector: jnp.ndarray = None,
               inverse: bool = False,
               with_translation: bool = False
               ):
    """Warp the given points using a warp field.

    Args:
      points: the points to warp.
      metadata: metadata indices if metadata_encoded is False else pre-encoded
        metadata.
      extra_params: A dictionary containing
        'alpha': the alpha value for the positional encoding.
      return_jacobian: if True compute and return the Jacobian of the warp.

    Returns:
      The warped points and the Jacobian of the warp if `return_jacobian` is
        True.
    """

    warped_points, screw_axis = self.warp(points,
                                          metadata,
                                          extra_params,
                                          return_screw=True,
                                          vector=vector,
                                          inverse=inverse,
                                          with_translation=with_translation
                                          )
    out = {
        'warped_points': warped_points,
        "screw_axis": screw_axis
    }

    if return_jacobian:
      jac_fn = jax.jacfwd(self.warp, argnums=0)
      warp_jacobian, _ = jac_fn(points, metadata, extra_params, False, vector, inverse)   # exclude jacobian of screw axis
      out['jacobian'] = warp_jacobian

    return out


@gin.configurable(denylist=['name'])
class BoneSE3Field(nn.Module):
  """Network that predicts warps as an SE(3) field.

  Attributes:
    points_encoder: the positional encoder for the points.
    metadata_encoder: an encoder for metadata.
    alpha: the alpha for the positional encoding.
    skips: the index of the layers with skip connections.
    depth: the depth of the network excluding the logit layer.
    hidden_channels: the width of the network hidden layers.
    activation: the activation for each layer.
    metadata_encoded: whether the metadata parameter is pre-encoded or not.
    hidden_initializer: the initializer for the hidden layers.
    output_initializer: the initializer for the last logit layer.
  """
  min_deg: int = 0
  max_deg: int = 8
  use_posenc_identity: bool = False

  activation: types.Activation = nn.relu
  norm: Optional[Any] = None
  skips: Iterable[int] = (4,)
  trunk_depth: int = 4
  trunk_width: int = 32
  rotation_depth: int = 0
  rotation_width: int = 128
  pivot_depth: int = 0
  pivot_width: int = 128
  translation_depth: int = 0
  translation_width: int = 128

  default_init: types.Initializer = jax.nn.initializers.xavier_uniform()
  rotation_init: types.Initializer = jax.nn.initializers.uniform(scale=1e-4)
  translation_init: types.Initializer = jax.nn.initializers.uniform(scale=1e-4)

  num_bones: int = 3
  moving_mlp_depth: int = 6
  moving_mlp_width: int = 128

  def setup(self):
    self.trunk = modules.MLP(
        depth=self.trunk_depth,
        width=self.trunk_width,
        hidden_activation=self.activation,
        hidden_norm=self.norm,
        hidden_init=self.default_init,
        skips=self.skips)

    branches = {
        'w':
            modules.MLP(
                depth=self.rotation_depth,
                width=self.rotation_width,
                hidden_activation=self.activation,
                hidden_norm=self.norm,
                hidden_init=self.default_init,
                output_init=self.rotation_init,
                output_channels=3),
        'v':
            modules.MLP(
                depth=self.translation_depth,
                width=self.translation_width,
                hidden_activation=self.activation,
                hidden_norm=self.norm,
                hidden_init=self.default_init,
                output_init=self.translation_init,
                output_channels=3),
    }

    # Note that this must be done this way instead of using mutable operations.
    # See https://github.com/google/flax/issues/524.
    self.branches = branches

    # bone parameters
    self.bone_centers = self.param('bone_centers', self.default_init, (self.num_bones, 3))
    self.bone_scales = self.param('bone_scales', self.default_init, (self.num_bones, 3))
    self.bone_quaternions = self.param('bone_quaternions', self.default_init, (self.num_bones, 4))
    self.moving_mlp = modules.MLP(
      depth=self.moving_mlp_depth,
      width=self.moving_mlp_width,
      hidden_activation=self.activation,
      hidden_norm=self.norm,
      hidden_init=self.default_init,
      output_init=self.translation_init,
      output_channels=1
    )

  def get_bone_se3(self,
                   metadata_embed: jnp.ndarray    # N x E
                   ):
    """
    use the mlp to query the se3 screw of the bones
    """
    # prepare bone index and warp embeddings
    N = metadata_embed.shape[0]

    bone_idx = jnp.eye(self.num_bones)    # B x B
    bone_idx = jnp.broadcast_to(bone_idx, [N * self.num_bones, self.num_bones])   # (N x B) x B
    metadata_embed = jnp.broadcast_to(metadata_embed, [N * self.num_bones, metadata_embed.shape[-1]])   # (N x B) x E
    inputs = jnp.concatenate([bone_idx, metadata_embed], axis=-1)   # (N x B) x (B + E)

    # query mlp
    trunk_output = self.trunk(inputs)
    w = self.branches['w'](trunk_output)    # (N x B) x 3
    v = self.branches['v'](trunk_output)    # (N x B) x 3

    return w, v

  def warp(self,
           points: jnp.ndarray,   # ... x 3
           w: jnp.ndarray,    # ... x 3
           v: jnp.ndarray,    # ... x 3
           rotation_only: bool = False,
           inverse: bool = False,
           return_transform: bool = False
           ):
    """
    Warp a 3D point/vector with given w and v
    if vector is none, transform points, else transform vector
    """
    theta = jnp.linalg.norm(w, axis=-1)
    w = w / theta[..., jnp.newaxis]
    v = v / theta[..., jnp.newaxis]
    screw_axis = jnp.concatenate([w, v], axis=-1)

    exp_se3_fn = jax.vmap(rigid.exp_se3, in_axes=(0, 0, None, None))
    transform = exp_se3_fn(screw_axis, theta, rotation_only, inverse)
    # transform = rigid.exp_se3(screw_axis, theta, rotation_only=rotation_only, inverse=inverse)

    homogenous_points = rigid.to_homogenous(points)[..., jnp.newaxis]
    warped_points = utils.matmul(transform, homogenous_points).squeeze(axis=-1)
    warped_points = rigid.from_homogenous(warped_points)

    if return_transform:
      return warped_points, transform
    else:
      return warped_points

  def warp_bones(self,
                 w: jnp.ndarray,    # (N x B) x 3
                 v: jnp.ndarray,    # (N x B) x 3
                 ):
    N = w.shape[0] // self.num_bones
    bone_centers = jnp.broadcast_to(self.bone_centers, [N * self.num_bones, 3])  # (N x B) x 3
    bone_scales = jnp.broadcast_to(self.bone_scales, [N * self.num_bones, 3])  # (N x B) x 3
    bone_quaternions = jnp.broadcast_to(self.bone_quaternions, [N * self.num_bones, 4])  # (N x B) x 4
    bone_rotations = to_rotation_matrix(bone_quaternions)
    #
    # # vmap the warping process
    # warp_fn = jax.vmap(self.warp, in_axes=(0, 0, 0, None, None, None))

    warped_bone_centers, transform = self.warp(bone_centers, w, v, return_transform=True)
    # warped_bone_centers, transform = warp_fn(bone_centers, w, v, False, False, True)
    warped_bone_scales = bone_scales
    rotation_matrix = transform[..., :3, :3]
    warped_bone_rotations = rotation_matrix @ bone_rotations
    return warped_bone_centers, warped_bone_scales, warped_bone_rotations

  def get_skinning_weights(self,
                           points: jnp.ndarray,         # N x 3
                           bone_centers: jnp.ndarray,   # (N x B) x 3
                           bone_scales: jnp.ndarray,    # (N x B) x 3
                           bone_rotations: jnp.ndarray  # (N x B) x 3 x 3
                           ):
    # N = points.shape[0]
    # bone_centers = bone_centers.reshape([N, -1, 3])   # N x B x 3
    # bone_scales = bone_scales.reshape([N, -1, 3])   # N x B x 3
    # bone_rotations = bone_rotations.reshape([N, -1, 3, 3])   # N x B x 3 x 3

    bone_probs = get_bone_probs_batch(points, bone_centers, bone_scales, bone_rotations)    # N x B
    bone_weights = nn.softmax(bone_probs, axis=-1)    # N x B

    return bone_weights

  def get_moving_mask(self,
                      points: jnp.ndarray,   # N x 3
                      metadata_embed: jnp.ndarray    # N x E
                      ):
    inputs = jnp.concatenate([points, metadata_embed], axis=-1)   # N x (3 + E)
    moving_mask = self.moving_mlp(inputs)   # N x 1
    # moving_mask = jnp.heaviside(moving_mask, 0)
    moving_mask = jax.nn.sigmoid(moving_mask)
    return moving_mask

  def __call__(self,
               points: jnp.ndarray,       # ... x 3
               metadata: jnp.ndarray,     # ... x E
               extra_params: Dict[str, Any],
               ):
    """Warp the given points using a warp field.

    Args:
      points: the points to warp.
      metadata: metadata indices if metadata_encoded is False else pre-encoded
        metadata.
      extra_params: A dictionary containing
        'alpha': the alpha value for the positional encoding.

    Returns:
      The warped points
    """
    if len(points.shape) == 1:
      points = points[jnp.newaxis, ...]
      metadata = metadata[jnp.newaxis, ...]
    batch_shape = points.shape[:-1]
    points = points.reshape([-1, points.shape[-1]])         # N x 3
    metadata = metadata.reshape([-1, metadata.shape[-1]])   # N x E
    N = points.shape[0]

    # get forward transforms
    w, v = self.get_bone_se3(metadata_embed=metadata)       # N x B x 3

    # get forward warped bones
    bone_centers, bone_scales, bone_rotations = self.warp_bones(w, v)

    # get skinning weights
    bone_weights = self.get_skinning_weights(points, bone_centers, bone_scales, bone_rotations)   # N x B

    # get backward warped points
    bone_points = jnp.broadcast_to(points, [N * self.num_bones, 3])   # (N x B) x 3
    warped_points = self.warp(bone_points, w, v, inverse=True)   # (N x B) x 3

    # weighted sum of warped points
    bone_weights = bone_weights[..., jnp.newaxis]   # N x B x 1
    warped_points = warped_points.reshape([N, self.num_bones, 3])
    warped_points = jnp.sum(bone_weights * warped_points, axis=1)    # N x 3

    # mask movement to separate moving and static points
    moving_mask = self.get_moving_mask(points, metadata)
    warped_points = moving_mask * warped_points + (1 - moving_mask) * points

    # restore shape
    if warped_points.shape[0] == 1:
      warped_points = jnp.squeeze(warped_points, axis=0)
      bone_weights = jnp.squeeze(bone_weights, axis=0)
      moving_mask = jnp.squeeze(moving_mask, axis=0)
    else:
      warped_points = warped_points.reshape([*batch_shape, 3])
      bone_weights = bone_weights.reshape([*batch_shape, self.num_bones])
      moving_mask = moving_mask.reshape([*batch_shape, 1])

    out = {
      'warped_points': warped_points,
      'bone_weights': bone_weights,
      'moving_mask': moving_mask
    }
    return out