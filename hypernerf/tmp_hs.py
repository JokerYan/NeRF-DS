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

"""Different model implementation plus a general port for all the models."""
import functools
import logging
from typing import Any, Callable, Dict, Optional, Tuple, Sequence, Mapping

import numpy as np
from flax import jax_utils
from flax import linen as nn
import gin
import immutabledict
import jax
from jax import lax
from jax import random
import jax.numpy as jnp

from hypernerf import model_utils
from hypernerf import modules
from hypernerf import types
# pylint: disable=unused-import
from hypernerf import warping
from hypernerf.model_utils import cal_ref_radiance


def filter_sigma(points, sigma, render_opts):
  """Filters the density based on various rendering arguments.

   - `dust_threshold` suppresses any sigma values below a threshold.
   - `bounding_box` suppresses any sigma values outside of a 3D bounding box.

  Args:
    points: the input points for each sample.
    sigma: the array of sigma values.
    render_opts: a dictionary containing any of the options listed above.

  Returns:
    A filtered sigma density field.
  """
  if render_opts is None:
    return sigma

  # Clamp densities below the set threshold.
  if 'dust_threshold' in render_opts:
    dust_thres = render_opts.get('dust_threshold', 0.0)
    sigma = (sigma >= dust_thres).astype(jnp.float32) * sigma
  if 'bounding_box' in render_opts:
    xmin, xmax, ymin, ymax, zmin, zmax = render_opts['bounding_box']
    render_mask = ((points[..., 0] >= xmin) & (points[..., 0] <= xmax)
                   & (points[..., 1] >= ymin) & (points[..., 1] <= ymax)
                   & (points[..., 2] >= zmin) & (points[..., 2] <= zmax))
    sigma = render_mask.astype(jnp.float32) * sigma

  return sigma

class CustomModel(nn.Module):
  pass

@gin.configurable(denylist=['name'])
class HyperSpecModel(CustomModel):
  """Nerf NN Model with both coarse and fine MLPs.

  Attributes:
    embeddings_dict: a dictionary containing the embeddings of each metadata
      key.
    use_viewdirs: bool, use viewdirs as a condition.
    noise_std: float, std dev of noise added to regularize sigma output.
    nerf_trunk_depth: int, the depth of the first part of MLP.
    nerf_trunk_width: int, the width of the first part of MLP.
    nerf_rgb_branch_depth: int, the depth of the second part of MLP.
    nerf_rgb_branch_width: int, the width of the second part of MLP.
    nerf_skips: which layers to add skip layers in the NeRF model.
    spatial_point_min_deg: min degree of positional encoding for positions.
    spatial_point_max_deg: max degree of positional encoding for positions.
    hyper_point_min_deg: min degree of positional encoding for hyper points.
    hyper_point_max_deg: max degree of positional encoding for hyper points.
    viewdir_min_deg: min degree of positional encoding for viewdirs.
    viewdir_max_deg: max degree of positional encoding for viewdirs.

    alpha_channels: int, the number of alpha_channelss.
    rgb_channels: int, the number of rgb_channelss.
    activation: the activation function used in the MLP.
    sigma_activation: the activation function applied to the sigma density.

    near: float, near clip.
    far: float, far clip.
    num_coarse_samples: int, the number of samples for coarse nerf.
    num_fine_samples: int, the number of samples for fine nerf.
    use_stratified_sampling: use stratified sampling.
    use_white_background: composite rendering on to a white background.
    use_linear_disparity: sample linearly in disparity rather than depth.

    use_nerf_embed: whether to use the template metadata.
    use_alpha_condition: whether to feed the appearance metadata to the alpha
      branch.
    use_rgb_condition: whether to feed the appearance metadata to the rgb
      branch.

    use_warp: whether to use the warp field or not.
    warp_metadata_config: the config for the warp metadata encoder.
    warp_min_deg: min degree of positional encoding for warps.
    warp_max_deg: max degree of positional encoding for warps.
  """
  embeddings_dict: Mapping[str, Sequence[int]] = gin.REQUIRED
  near: float = gin.REQUIRED
  far: float = gin.REQUIRED

  # NeRF architecture.
  use_viewdirs: bool = True
  noise_std: Optional[float] = None
  nerf_trunk_depth: int = 8
  nerf_trunk_width: int = 256
  nerf_rgb_branch_depth: int = 1
  nerf_rgb_branch_width: int = 128
  nerf_skips: Tuple[int] = (4,)

  # NeRF rendering.
  num_coarse_samples: int = 196
  num_fine_samples: int = 196
  use_stratified_sampling: bool = True
  use_white_background: bool = False
  use_linear_disparity: bool = False
  use_sample_at_infinity: bool = True

  spatial_point_min_deg: int = 0
  spatial_point_max_deg: int = 10
  hyper_point_min_deg: int = 0
  hyper_point_max_deg: int = 4
  viewdir_min_deg: int = 0
  viewdir_max_deg: int = 4
  use_posenc_identity: bool = True

  alpha_channels: int = 1
  rgb_channels: int = 3
  activation: types.Activation = nn.relu
  norm_type: Optional[str] = None
  sigma_activation: types.Activation = nn.softplus

  # NeRF metadata configs.
  use_nerf_embed: bool = False
  nerf_embed_cls: Callable[..., nn.Module] = (
    functools.partial(modules.GLOEmbed, num_dims=8))
  nerf_embed_key: str = 'appearance'
  use_alpha_condition: bool = False
  use_rgb_condition: bool = False
  hyper_slice_method: str = 'none'
  hyper_embed_cls: Callable[..., nn.Module] = (
    functools.partial(modules.GLOEmbed, num_dims=8))

  # hyper_embed_key: str = 'appearance'
  hyper_embed_key: str = 'warp'
  # TODO: change this key from 'appearance' to 'warp'
  """
  All current experiments have hyper_use_warp_embed turned on. This means that the hyper_embed_key is not used.
  In our future double camera datasets, there will only be two appearance code, one for each camera.
  In that case, the hyper embedding cannot use 'appearance' as the embedding key in all circumstances.
  The best solution is the change this to a separate 'hyper' key, but it might cause errors on old dataset if
    hyper_use_warp_embed switched off.
  """
  hyper_use_warp_embed: bool = True
  hyper_sheet_mlp_cls: Callable[..., nn.Module] = modules.HyperSheetMLP
  hyper_sheet_use_input_points: bool = True

  # Warp configs.
  use_warp: bool = False
  warp_field_cls: Callable[..., nn.Module] = warping.SE3Field
  warp_embed_cls: Callable[..., nn.Module] = (
    functools.partial(modules.GLOEmbed, num_dims=8))
  warp_embed_key: str = 'warp'

  # Spec config
  predict_norm: bool = False
  norm_supervision_type: str = 'warped'   # warped, canonical direct
  stop_norm_gradient: bool = True
  norm_input_min_deg: int = 0
  norm_input_max_deg: int = 4
  use_viewdirs_in_hyper: bool = False
  use_x_in_rgb_condition: bool = False

  # Warp c config
  use_warp_c: bool = True
  warp_c_field_cls: Callable[..., nn.Module] = warping.SE3Field
  warp_c_embed_cls: Callable[..., nn.Module] = (
    functools.partial(modules.GLOEmbed, num_dims=8)
  )
  warp_c_embed_key: str = 'warp'

  # Hyper c config
  use_hyper_c: bool = False
  hyper_c_hyper_input: bool = False
  use_hyper_c_embed: bool = True
  hyper_c_use_warp_c_embed: bool = True
  hyper_c_embed_cls: Callable[..., nn.Module] = (
    functools.partial(modules.GLOEmbed, num_dims=8)
  )
  hyper_c_mlp_cls: Callable[..., nn.Module] = modules.HyperSheetMLP
  hyper_c_num_dims: int = 2

  # norm voxel
  use_norm_voxel: bool = False

  # reflected radiance
  use_ref_radiance: bool = False

  # color mixed from diffuse and specular
  mix_d_s_color: bool = False

  @property
  def num_nerf_embeds(self):
    return max(self.embeddings_dict[self.nerf_embed_key]) + 1

  @property
  def num_warp_embeds(self):
    return max(self.embeddings_dict[self.warp_embed_key]) + 1

  @property
  def num_warp_c_embeds(self):
    return max(self.embeddings_dict[self.warp_c_embed_key]) + 1

  @property
  def num_hyper_embeds(self):
    return max(self.embeddings_dict[self.hyper_embed_key]) + 1

  @property
  def nerf_embeds(self):
    return jnp.array(self.embeddings_dict[self.nerf_embed_key], jnp.uint32)

  @property
  def warp_embeds(self):
    return jnp.array(self.embeddings_dict[self.warp_embed_key], jnp.uint32)

  @property
  def hyper_embeds(self):
    return jnp.array(self.embeddings_dict[self.hyper_embed_key], jnp.uint32)

  @property
  def has_hyper(self):
    """Whether the model uses a separate hyper embedding."""
    return self.hyper_slice_method != 'none'

  @property
  def has_hyper_embed(self):
    """Whether the model uses a separate hyper embedding."""
    # If the warp field outputs the hyper coordinates then there is no separate
    # hyper embedding.
    return self.has_hyper

  @property
  def has_embeds(self):
    return self.has_hyper_embed or self.use_warp or self.use_nerf_embed

  @staticmethod
  def _encode_embed(embed, embed_fn):
    """Encodes embeddings.

    If the channel size 1, it is just a single metadata ID.
    If the channel size is 3:
      the first channel is the left metadata ID,
      the second channel is the right metadata ID,
      the last channel is the progression from left to right (between 0 and 1).

    Args:
      embed: a (*, 1) or (*, 3) array containing metadata.
      embed_fn: the embedding function.

    Returns:
      A (*, C) array containing encoded embeddings.
    """
    if embed.shape[-1] == 3:
      left, right, progression = jnp.split(embed, 3, axis=-1)
      left = embed_fn(left.astype(jnp.uint32))
      right = embed_fn(right.astype(jnp.uint32))
      return (1.0 - progression) * left + progression * right
    else:
      return embed_fn(embed)

  def encode_hyper_embed(self, metadata):
    if self.hyper_slice_method == 'axis_aligned_plane':
      if self.hyper_use_warp_embed:
        return self._encode_embed(metadata[self.warp_embed_key],
                                  self.warp_embed)
      else:
        return self._encode_embed(metadata[self.hyper_embed_key],
                                  self.hyper_embed)
    elif self.hyper_slice_method == 'bendy_sheet':
      # The bendy sheet shares the metadata of the warp.
      if self.hyper_use_warp_embed:
        return self._encode_embed(metadata[self.warp_embed_key],
                                  self.warp_embed)
      else:
        return self._encode_embed(metadata[self.hyper_embed_key],
                                  self.hyper_embed)
    else:
      raise RuntimeError(
        f'Unknown hyper slice method {self.hyper_slice_method}.')

  def encode_nerf_embed(self, metadata):
    return self._encode_embed(metadata[self.nerf_embed_key], self.nerf_embed)

  def encode_warp_embed(self, metadata):
    return self._encode_embed(metadata[self.warp_embed_key], self.warp_embed)

  def setup(self):
    if (self.use_nerf_embed
            and not (self.use_rgb_condition
                     or self.use_alpha_condition)):
      raise ValueError('Template metadata is enabled but none of the condition'
                       'branches are.')

    if self.use_nerf_embed:
      self.nerf_embed = self.nerf_embed_cls(num_embeddings=self.num_nerf_embeds)
    if self.use_warp:
      self.warp_embed = self.warp_embed_cls(num_embeddings=self.num_warp_embeds)
    if self.use_warp_c:
      self.warp_c_embed = self.warp_c_embed_cls(num_embeddings=self.num_warp_c_embeds)

    if self.hyper_slice_method == 'axis_aligned_plane':
      self.hyper_embed = self.hyper_embed_cls(
        num_embeddings=self.num_hyper_embeds)
    elif self.hyper_slice_method == 'bendy_sheet':
      if not self.hyper_use_warp_embed:
        self.hyper_embed = self.hyper_embed_cls(
          num_embeddings=self.num_hyper_embeds)
      self.hyper_sheet_mlp = self.hyper_sheet_mlp_cls()

    if self.use_hyper_c:
      if not self.hyper_c_use_warp_c_embed:
        self.hyper_c_embed = self.hyper_c_embed_cls(
          num_embeddings=self.num_hyper_embeds
        )
      self.hyper_c_mlp = self.hyper_c_mlp_cls()

    if self.use_warp:
      self.warp_field = self.warp_field_cls()
    if self.use_warp_c:
      self.warp_c_field = self.warp_c_field_cls()

    norm_layer = modules.get_norm_layer(self.norm_type)
    nerf_mlps = {
      'coarse': modules.NerfMLP(
        trunk_depth=self.nerf_trunk_depth,
        trunk_width=self.nerf_trunk_width,
        rgb_branch_depth=self.nerf_rgb_branch_depth,
        rgb_branch_width=self.nerf_rgb_branch_width,
        activation=self.activation,
        norm=norm_layer,
        skips=self.nerf_skips,
        alpha_channels=self.alpha_channels,
        rgb_channels=self.rgb_channels,
        predict_norm=self.predict_norm,
        predict_d_color=self.mix_d_s_color
      )
    }
    if self.num_fine_samples > 0:
      nerf_mlps['fine'] = modules.NerfMLP(
        trunk_depth=self.nerf_trunk_depth,
        trunk_width=self.nerf_trunk_width,
        rgb_branch_depth=self.nerf_rgb_branch_depth,
        rgb_branch_width=self.nerf_rgb_branch_width,
        activation=self.activation,
        norm=norm_layer,
        skips=self.nerf_skips,
        alpha_channels=self.alpha_channels,
        rgb_channels=self.rgb_channels,
        predict_norm=self.predict_norm,
        predict_d_color=self.mix_d_s_color
      )
    self.nerf_mlps = nerf_mlps

    # # surface normal voxels
    # self.surface_norm_voxels = jnp.zeros((500, 100, 100, 100), dtype=jnp.float16)
    if self.use_norm_voxel:
      self.norm_voxel = modules.NormVoxels()

  def get_condition_inputs(self, metadata, metadata_encoded=False):
    """Create the condition inputs for the NeRF template."""
    alpha_conditions = []
    rgb_conditions = []

    # No view dirs in rgb_condition
    # if self.use_viewdirs and not self.use_hyper_c:
    #   viewdirs_feat = model_utils.posenc(
    #     viewdirs,
    #     min_deg=self.viewdir_min_deg,
    #     max_deg=self.viewdir_max_deg,
    #     use_identity=self.use_posenc_identity)
    #   rgb_conditions.append(viewdirs_feat)

    if self.use_nerf_embed:
      if metadata_encoded:
        nerf_embed = metadata['encoded_nerf']
      else:
        nerf_embed = metadata[self.nerf_embed_key]
        nerf_embed = self.nerf_embed(nerf_embed)
      if self.use_alpha_condition:
        alpha_conditions.append(nerf_embed)
      if self.use_rgb_condition:
        rgb_conditions.append(nerf_embed)

    # The condition inputs have a shape of (B, C) now rather than (B, S, C)
    # since we assume all samples have the same condition input. We might want
    # to change this later.
    alpha_conditions = (
      jnp.concatenate(alpha_conditions, axis=-1)
      if alpha_conditions else None)
    rgb_conditions = (
      jnp.concatenate(rgb_conditions, axis=-1)
      if rgb_conditions else None)
    return alpha_conditions, rgb_conditions

  def pre_process_query(self,
                        points,
                        metadata,
                        extra_params,
                        metadata_encoded=False, ):
    alpha_condition, rgb_condition = (
      self.get_condition_inputs(metadata, metadata_encoded))

    points_feat = model_utils.posenc(
      points[..., :3],
      min_deg=self.spatial_point_min_deg,
      max_deg=self.spatial_point_max_deg,
      use_identity=self.use_posenc_identity,
      alpha=extra_params['nerf_alpha'])
    # Encode hyper-points if present.
    if points.shape[-1] > 3:
      hyper_feats = model_utils.posenc(
        points[..., 3:],
        min_deg=self.hyper_point_min_deg,
        max_deg=self.hyper_point_max_deg,
        use_identity=False,
        alpha=extra_params['hyper_alpha'])
      points_feat = jnp.concatenate([points_feat, hyper_feats], axis=-1)

    if len(points_feat.shape) > 1:
      num_samples = points_feat.shape[1]
    else:
      num_samples = 1

    return points_feat, alpha_condition, rgb_condition, num_samples

  def query_template_bottleneck(self,
                                level,
                                points_feat,
                                alpha_condition,
                                rgb_condition
                                ):
    trunk_output, bottleneck = self.nerf_mlps[level].query_bottleneck(points_feat, alpha_condition, rgb_condition)
    return trunk_output, bottleneck

  def query_template_sigma(self,
                           level,
                           trunk_output,
                           bottleneck,
                           alpha_condition
                           ):
    sigma, norm, d_color, s_weight = self.nerf_mlps[level].query_sigma(trunk_output, bottleneck, alpha_condition)
    return sigma, norm, d_color, s_weight

  def query_template_s_color(self,
                             level,
                             trunk_output,
                             bottleneck,
                             rgb_condition,
                             extra_rgb_condition=None
                             ):

    s_color = self.nerf_mlps[level].query_rgb(trunk_output, bottleneck, rgb_condition,
                                              extra_rgb_condition=extra_rgb_condition)
    return s_color

  def post_process_query(self, level, rgb, sigma, num_samples):
    raw = {
      'rgb': rgb.reshape((-1, num_samples, self.rgb_channels)),
      'alpha': sigma.reshape((-1, num_samples, self.alpha_channels)),
    }

    raw = model_utils.noise_regularize(
      self.make_rng(level), raw, self.noise_std, self.use_stratified_sampling)

    rgb = nn.sigmoid(raw['rgb'])
    sigma = self.sigma_activation(jnp.squeeze(raw['alpha'], axis=-1))

    return rgb, sigma

  def map_vectors(self, points, vectors, warp_embed, extra_params, return_warp_jacobian=False, inverse=False, with_translation=False):
    warp_jacobian = None
    screw_axis = None
    if self.use_warp:
      if len(vectors.shape) > 1:
        warp_fn = jax.vmap(jax.vmap(self.warp_field, in_axes=(0, 0, None, None, 0, None, None)),
                           in_axes=(0, 0, None, None, 0, None, None))
      else:
        warp_fn = self.warp_field
      warp_out = warp_fn(points,
                         warp_embed,
                         extra_params,
                         return_warp_jacobian,
                         vectors,   # rotation only,
                         inverse,
                         with_translation
                         )
      if return_warp_jacobian:
        warp_jacobian = warp_out['jacobian']
      warped_points = warp_out['warped_points']
      screw_axis = warp_out['screw_axis']
    else:
      warped_points = vectors

    return warped_points, warp_jacobian, screw_axis

  def map_spatial_points(self, points, warp_embed, extra_params, use_warp=True,
                         return_warp_jacobian=False):
    warp_jacobian = None
    screw_axis = None
    if self.use_warp and use_warp:
      if len(points.shape) > 1:
        warp_fn = jax.vmap(jax.vmap(self.warp_field, in_axes=(0, 0, None, None)),
                           in_axes=(0, 0, None, None))
      else:
        warp_fn = self.warp_field
      warp_out = warp_fn(points,
                         warp_embed,
                         extra_params,
                         return_warp_jacobian)
      if return_warp_jacobian:
        warp_jacobian = warp_out['jacobian']
      warped_points = warp_out['warped_points']
      screw_axis = warp_out['screw_axis']
    else:
      warped_points = points

    return warped_points, warp_jacobian, screw_axis

  def map_hyper_points(self, points, hyper_embed, extra_params,
                       hyper_point_override=None):
    """Maps input points to hyper points.

    Args:
      points: the input points.
      hyper_embed: the hyper embeddings.
      extra_params: extra params to pass to the slicing MLP if applicable.
      hyper_point_override: this may contain an override for the hyper points.
        Useful for rendering at specific hyper dimensions.

    Returns:
      An array of hyper points.
    """
    if hyper_point_override is not None:
      hyper_points = jnp.broadcast_to(
        hyper_point_override[:, None, :],
        (*points.shape[:-1], hyper_point_override.shape[-1]))
    elif self.hyper_slice_method == 'axis_aligned_plane':
      hyper_points = hyper_embed
    elif self.hyper_slice_method == 'bendy_sheet':
      hyper_points = self.hyper_sheet_mlp(
        points,
        hyper_embed,
        alpha=extra_params['hyper_sheet_alpha'])
    else:
      return None

    return hyper_points

  def map_points(self, points, warp_embed, hyper_embed, viewdirs, extra_params,
                 use_warp=True, return_warp_jacobian=False, return_hyper_jacobian=False,
                 hyper_point_override=None):
    """Map input points to warped spatial and hyper points.

    Args:
      points: the input points to warp.
      warp_embed: the warp embeddings.
      hyper_embed: the hyper embeddings.
      extra_params: extra parameters to pass to the warp field/hyper field.
      use_warp: whether to use the warp or not.
      return_warp_jacobian: whether to return the warp jacobian or not.
      hyper_point_override: this may contain an override for the hyper points.
        Useful for rendering at specific hyper dimensions.

    Returns:
      A tuple containing `(warped_points, warp_jacobian)`.
    """
    # Map input points to warped spatial and hyper points.
    spatial_points, warp_jacobian, screw_axis = self.map_spatial_points(
      points, warp_embed, extra_params, use_warp=use_warp,
      return_warp_jacobian=return_warp_jacobian)
    if self.use_viewdirs_in_hyper:
      raise NotImplementedError
      # if len(hyper_embed.shape) == 3:
      #   num_samples = hyper_embed.shape[1]
      #   viewdirs = jnp.tile(jnp.expand_dims(viewdirs, 1), [1, num_samples, 1])
      # hyper_embed = jnp.concatenate([hyper_embed, viewdirs], axis=-1)
    if return_hyper_jacobian:
      hyper_points = self.map_hyper_points(
        points, hyper_embed, extra_params,
        # Override hyper points if present in metadata dict.
        hyper_point_override=hyper_point_override)
      # jacobian wrt the hyper embedding
      hyper_jacobian = jax.jacrev(self.map_hyper_points, argnums=1)(
        points, hyper_embed, extra_params, hyper_point_override
      )
    else:
      hyper_points = self.map_hyper_points(
        points, hyper_embed, extra_params,
        # Override hyper points if present in metadata dict.
        hyper_point_override=hyper_point_override)
      hyper_jacobian = None

    if hyper_points is not None:
      warped_points = jnp.concatenate([spatial_points, hyper_points], axis=-1)
    else:
      warped_points = spatial_points

    return warped_points, warp_jacobian, hyper_jacobian, screw_axis

  def map_spatial_c(self, points, warp_c_embed, viewdirs_feat, norm_input_feat, ref_radiance_feat,
                    extra_params, return_warp_c_jacobian=False):
    """
      please note that points, viewdirs, norm_input, ref_radiance will be going through positional
      encoding later in the mlp. So no need to posenc them first.
      hyper_c_embed, however, will not be pos-encoded.

    """
    num_samples = points.shape[1]
    viewdirs_feat_expanded = jnp.tile(jnp.expand_dims(viewdirs_feat, axis=1), [1, num_samples, 1])

    warp_c_embed = jnp.concatenate([warp_c_embed, viewdirs_feat_expanded], axis=-1)
    if norm_input_feat is not None:
      warp_c_embed = jnp.concatenate([warp_c_embed, norm_input_feat], axis=-1)
    if ref_radiance_feat is not None:
      warp_c_embed = jnp.concatenate([warp_c_embed, ref_radiance_feat], axis=-1)

    warp_c_jacobian = None
    if self.use_warp_c:
      if len(points.shape) > 1:
        warp_fn = jax.vmap(jax.vmap(self.warp_c_field, in_axes=(0, 0, None, None)),
                           in_axes=(0, 0, None, None))
      else:
        warp_fn = self.warp_c_field
      warp_out = warp_fn(points,
                         warp_c_embed,
                         extra_params,
                         return_warp_c_jacobian)
      if return_warp_c_jacobian:
        warp_c_jacobian = warp_out['jacobian']
      warped_c_points = warp_out['warped_points']
    else:
      warped_c_points = points

    return warped_c_points, warp_c_jacobian

  def map_hyper_c(self, points, hyper_c_embed, viewdirs, norm_input, ref_radiance, return_hyper_c_jacobian=False):
    """
      please note that points, viewdirs, norm_input, ref_radiance will be going through positional
      encoding later in the mlp. So no need to posenc them first.
      hyper_c_embed, however, will not be pos-encoded.

      Please note that when use_hyper_c_embed is turned off, the hyper_c_embed is not passed into the mlp,
      hence 'block t'.
    """
    num_samples = points.shape[1]
    viewdirs_expanded = jnp.tile(jnp.expand_dims(viewdirs, axis=1), [1, num_samples, 1])

    hyper_c_input = jnp.concatenate([points, viewdirs_expanded], axis=-1)
    if norm_input is not None:
      hyper_c_input = jnp.concatenate([hyper_c_input, norm_input], axis=-1)
    if ref_radiance is not None:
      hyper_c_input = jnp.concatenate([hyper_c_input, ref_radiance], axis=-1)

    def query_hyper_c_mlp(hyper_c_input, hyper_c_embed):
      hyper_c = self.hyper_c_mlp(
        hyper_c_input, hyper_c_embed, alpha=None, use_embed=self.use_hyper_c_embed, output_channel=self.hyper_c_num_dims
      )
      return hyper_c

    def query_hyper_c_and_jacobian(hyper_c_input, hyper_c_embed):
      hyper_c = query_hyper_c_mlp(hyper_c_input, hyper_c_embed)
      if return_hyper_c_jacobian:
        hyper_c_jacobian = jax.jacrev(query_hyper_c_mlp, argnums=1)(hyper_c_input, hyper_c_embed)
      else:
        hyper_c_jacobian = None
      return hyper_c, hyper_c_jacobian

    hyper_c_fn = jax.vmap(jax.vmap(query_hyper_c_and_jacobian, in_axes=(0, 0)), in_axes=(0, 0))
    hyper_c, hyper_c_jacobian = hyper_c_fn(hyper_c_input, hyper_c_embed)

    assert hyper_c.shape[-1] == self.hyper_c_num_dims, (hyper_c.shape, self.hyper_c_num_dims)
    return hyper_c, hyper_c_jacobian

  def apply_warp(self, points, warp_embed, extra_params):
    warp_embed = self.warp_embed(warp_embed)
    return self.warp_field(points, warp_embed, extra_params)

  def apply_warp_c(self, points, warp_c_embed, extra_params):
    warp_c_embed = self.warp_c_embed(warp_c_embed)
    return self.warp_c_field(points, warp_c_embed, extra_params)

  def render_samples(self,
                     level,
                     points,
                     z_vals,
                     directions,
                     viewdirs,
                     metadata,
                     extra_params,
                     use_warp=True,
                     metadata_encoded=False,
                     return_warp_jacobian=False,
                     return_hyper_jacobian=False,
                     return_hyper_c_jacobian=False,
                     return_nv_details=True,
                     use_sample_at_infinity=False,
                     render_opts=None,
                     screw_input_mode=None,
                     use_sigma_gradient=False,
                     use_predicted_norm=False,
                     norm_voxel_lr=0,
                     norm_voxel_ratio=1,
                     ):
    out = {'points': points}

    batch_shape = points.shape[:-1]
    # Create the warp embedding.
    if use_warp:
      if metadata_encoded:
        warp_embed = metadata['encoded_warp']
      else:
        warp_embed = metadata[self.warp_embed_key]
        warp_embed = self.warp_embed(warp_embed)
    else:
      warp_embed = None

    if self.use_warp_c:
      warp_c_embed = metadata[self.warp_c_embed_key]
      warp_c_embed = self.warp_c_embed(warp_c_embed)
    else:
      warp_c_embed = None

    # Create the hyper embedding.
    if self.has_hyper_embed:
      if metadata_encoded:
        hyper_embed = metadata['encoded_hyper']
      elif self.hyper_use_warp_embed:
        hyper_embed = warp_embed
      else:
        hyper_embed = metadata[self.hyper_embed_key]
        hyper_embed = self.hyper_embed(hyper_embed)
    else:
      hyper_embed = None

    if self.use_hyper_c:
      if self.hyper_c_use_warp_c_embed:
        hyper_c_embed = warp_c_embed
      else:
        hyper_c_embed = metadata[self.hyper_embed_key]  # use hyper embed id
        hyper_c_embed = self.hyper_c_embed(hyper_c_embed)
    else:
      hyper_c_embed = None

    # Broadcast embeddings.
    if warp_embed is not None:
      warp_embed = jnp.broadcast_to(
        warp_embed[:, jnp.newaxis, :],
        shape=(*batch_shape, warp_embed.shape[-1]))
    if hyper_embed is not None:
      hyper_embed = jnp.broadcast_to(
        hyper_embed[:, jnp.newaxis, :],
        shape=(*batch_shape, hyper_embed.shape[-1]))
    if warp_c_embed is not None:
      warp_c_embed = jnp.broadcast_to(
        warp_c_embed[:, jnp.newaxis, :],
        shape=(*batch_shape, warp_embed.shape[-1]))
    if hyper_c_embed is not None:
      hyper_c_embed = jnp.broadcast_to(
        hyper_c_embed[:, jnp.newaxis, :],
        shape=(*batch_shape, hyper_c_embed.shape[-1]))

    if self.predict_norm and self.norm_supervision_type == 'canonical':
      raise NotImplementedError

    def cal_single_pt_sigma(points, warp_embed, hyper_embed, viewdirs):
      # Map input points to warped spatial and hyper points.
      warped_points, warp_jacobian, hyper_jacobian, screw_axis = self.map_points(
        points, warp_embed, hyper_embed, viewdirs, extra_params, use_warp=use_warp,
        return_warp_jacobian=return_warp_jacobian, return_hyper_jacobian=return_hyper_jacobian,
        # Override hyper points if present in metadata dict.
        hyper_point_override=metadata.get('hyper_point'))

      points_feat, alpha_condition, rgb_condition, num_samples = self.pre_process_query(warped_points,
                                                                                        metadata,
                                                                                        extra_params, metadata_encoded)
      trunk_output, bottleneck = self.query_template_bottleneck(level, points_feat, alpha_condition, rgb_condition)
      sigma, norm, d_color, s_weight = self.query_template_sigma(level, trunk_output, bottleneck, alpha_condition)
      aux_output = {
        'norm': norm,
        'd_color': d_color,
        's_weight': s_weight,
        'warped_points': warped_points,
        'warp_jacobian': warp_jacobian,
        'hyper_jacobian': hyper_jacobian,
        'screw_axis': screw_axis,
        'trunk_output': trunk_output,
        'alpha_condition': alpha_condition,
        'rgb_condition': rgb_condition,
        'bottleneck': bottleneck,
      }
      sigma = jnp.squeeze(sigma)
      return sigma, aux_output

    def cal_sigma_gradient(points, warp_embed, hyper_embed, viewdirs):
      value, gradient = jax.value_and_grad(cal_single_pt_sigma, argnums=0, has_aux=True)(points, warp_embed, hyper_embed, viewdirs)
      return value, - gradient

    sigma_gradient_fn = jax.vmap(jax.vmap(cal_sigma_gradient, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0))
    (sigma, aux_output), sigma_gradient = sigma_gradient_fn(points, warp_embed, hyper_embed, viewdirs)
    sigma_gradient = jnp.squeeze(sigma_gradient)

    # normalize
    sigma_gradient = model_utils.normalize_vector(sigma_gradient)
    # out['sigma_gradient'] = sigma_gradient
    # out['sigma_gradient'] = sigma_gradient_w

    # flatten except for feature dim, when applicable
    num_samples = points.shape[1]
    for key in aux_output.keys():
      if aux_output[key] is not None and len(aux_output[key].shape) > 2:
        if key == 'hyper_jacobian':
          continue
        aux_output[key] = jnp.reshape(aux_output[key], (-1, aux_output[key].shape[-1]))

    # unpack aux_output
    norm = aux_output['norm']
    d_color = aux_output['d_color']
    s_weight = aux_output['s_weight']
    warped_points = aux_output['warped_points']
    warp_jacobian = aux_output['warp_jacobian']
    hyper_jacobian = aux_output['hyper_jacobian']
    screw_axis = aux_output['screw_axis']
    trunk_output = aux_output['trunk_output']
    alpha_condition = aux_output['alpha_condition']
    rgb_condition = aux_output['rgb_condition']
    bottleneck = aux_output['bottleneck']

    # rgb
    if norm is not None:
      norm = jnp.reshape(norm, (-1, num_samples, norm.shape[-1]))

    if self.use_norm_voxel:
      # propogate time to all points on the same ray, (Ray, 1) -> (N, )
      time_array = jnp.array(metadata[self.warp_embed_key])
      points_per_ray = points.shape[1]
      time_array = jnp.tile(time_array, [1, points_per_ray])
      flat_time = time_array.reshape([-1])

      # flatten all points, (Ray, Sample, 3) -> (N, 3)
      flat_points = points.reshape([-1, 3])

      inter_norm, nv_vertex_values, nv_vertex_coef = self.norm_voxel.get_interpolation_value(flat_time, flat_points)

      ray_count, sample_count, _ = points.shape
      inter_norm = inter_norm.reshape([ray_count, sample_count, 3])
      inter_norm = model_utils.normalize_vector(inter_norm)

      nv_vertex_values = nv_vertex_values.reshape([ray_count, sample_count, 8, 3])
      nv_vertex_coef = nv_vertex_coef.reshape([ray_count, sample_count, 8])

    if self.use_norm_voxel:
      # weight inter_norm with pred_norm according to the norm_voxel_ratio
      weighted_norm = norm_voxel_ratio * inter_norm + (1 - norm_voxel_ratio) * norm

      if self.stop_norm_gradient:
        norm_input = lax.stop_gradient(weighted_norm)
      else:
        norm_input = weighted_norm
    elif use_sigma_gradient:
      assert not use_predicted_norm
      if self.stop_norm_gradient:
        norm_input = lax.stop_gradient(sigma_gradient)
      else:
        norm_input = sigma_gradient
    elif use_predicted_norm:
      assert not use_sigma_gradient
      # transform from canonical space to observation space
      normalized_norm = model_utils.normalize_vector(norm)
      if self.norm_supervision_type == 'warped' or self.norm_supervision_type == 'canonical':
        inverse_norm, _, _ = self.map_vectors(points, normalized_norm, warp_embed, extra_params, inverse=True)
        norm_input = inverse_norm
      elif self.norm_supervision_type == 'direct':
        norm_input = norm
      else:
        raise NotImplementedError
      if self.stop_norm_gradient:
        norm_input = lax.stop_gradient(norm_input)
    else:
      norm_input = None

    if norm_input is not None:
      norm_input = model_utils.normalize_vector(norm_input)
      norm_input_feat = model_utils.posenc(
        norm_input,
        min_deg=self.norm_input_min_deg,
        max_deg=self.norm_input_max_deg,
        use_identity=self.use_posenc_identity,
        alpha=extra_params['norm_input_alpha'])
    else:
      norm_input_feat = None

    ref_radiance = None
    ref_radiance_feat = None
    if self.use_ref_radiance:
      assert norm_input is not None
      viewdirs_expanded = jnp.tile(jnp.expand_dims(viewdirs, axis=1), [1, num_samples, 1])
      ref_radiance = cal_ref_radiance(viewdirs_expanded, norm_input)
      ref_radiance_feat = model_utils.posenc(
        ref_radiance,
        min_deg=self.norm_input_min_deg,
        max_deg=self.norm_input_max_deg,
        use_identity=self.use_posenc_identity,
        alpha=extra_params['norm_input_alpha'])

    warped_c_feat = None
    if self.use_warp_c:
      viewdirs_feat = model_utils.posenc(
        viewdirs,
        min_deg=self.viewdir_min_deg,
        max_deg=self.viewdir_max_deg,
        use_identity=self.use_posenc_identity)
      warped_c_points, _ = self.map_spatial_c(points, warp_c_embed, viewdirs_feat, norm_input_feat,
                                              ref_radiance_feat, extra_params, return_warp_c_jacobian=False)
      color_delta_x = lax.stop_gradient(warped_c_points - points)
      warped_c_points = jnp.reshape(warped_c_points, [-1, warped_c_points.shape[-1]])
      warped_c_feat = model_utils.posenc(
        warped_c_points[..., 3:],
        min_deg=self.spatial_point_min_deg,
        max_deg=self.spatial_point_max_deg,
        use_identity=self.use_posenc_identity,
        alpha=extra_params['nerf_alpha']
      )

    hyper_c_feat = None
    if self.use_hyper_c:
      if self.hyper_c_hyper_input:
        points_input = lax.stop_gradient(warped_points)
        points_input = jnp.reshape(points_input, [-1, num_samples, points_input.shape[-1]])
      else:
        points_input = lax.stop_gradient(points)
      # hyper_c_embed may not be used, depending on the use_hyper_c_embed settings
      hyper_c, hyper_c_jacobian = self.map_hyper_c(points_input, hyper_c_embed, viewdirs,
                                                   norm_input, ref_radiance, return_hyper_c_jacobian)

      hyper_c = jnp.reshape(hyper_c, [-1, hyper_c.shape[-1]])
      hyper_c_feat = model_utils.posenc(
        hyper_c,
        min_deg=self.hyper_point_min_deg,
        max_deg=self.hyper_point_max_deg,
        use_identity=False,
        alpha=extra_params['hyper_alpha'])

    extra_rgb_condition = None
    if self.use_warp_c:
      pass
      # DEBUG
      # extra_rgb_condition = warped_c_feat
    if self.use_hyper_c:
      if extra_rgb_condition is None:
        extra_rgb_condition = hyper_c_feat
      else:
        extra_rgb_condition = jnp.concatenate([extra_rgb_condition, hyper_c_feat], axis=-1)

    s_color = self.query_template_s_color(level, trunk_output, bottleneck, rgb_condition, extra_rgb_condition)

    if self.mix_d_s_color:
      # combine s color with d color
      s_weight = nn.sigmoid(s_weight)
      rgb = (1 - s_weight) * d_color + s_weight * s_color
    else:
      rgb = s_color

    rgb, sigma = self.post_process_query(level, rgb, sigma, num_samples)
    out['sigma'] = sigma

    # transform norm from observation to canonical
    sigma_gradient_r, _, _ = self.map_vectors(points, sigma_gradient, warp_embed, extra_params)
    sigma_gradient_r = model_utils.normalize_vector(sigma_gradient_r)

    # Filter densities based on rendering options.
    sigma = filter_sigma(points, sigma, render_opts)

    # visualize R
    rotation_reference = jnp.ones_like(points)
    rotation_reference = model_utils.normalize_vector(rotation_reference)
    rotation_field, _, _ = self.map_vectors(points, rotation_reference, warp_embed, extra_params)
    rotation_field = rotation_field[..., :3]
    rotation_field = model_utils.normalize_vector(rotation_field)

    # visualize t
    translation_reference = jnp.zeros_like(points)
    translation_field, _, _ = self.map_vectors(points, translation_reference, warp_embed, extra_params, with_translation=True)
    translation_field = translation_field[..., :3]

    warped_points = jnp.reshape(warped_points, (-1, num_samples, warped_points.shape[-1]))
    if warp_jacobian is not None:
      warp_jacobian = jnp.reshape(warp_jacobian, (-1, num_samples, warp_jacobian.shape[-1]))
      out['warp_jacobian'] = warp_jacobian
    out['warped_points'] = warped_points
    out.update(model_utils.volumetric_rendering(
      rgb,
      sigma,
      z_vals,
      directions,
      use_white_background=self.use_white_background,
      sample_at_infinity=use_sample_at_infinity))

    # calculate surface norm consistency
    if self.predict_norm:
      norm = jnp.reshape(norm, (-1, num_samples, norm.shape[-1]))
      out['predicted_norm'] = norm
      if self.norm_supervision_type == 'warped':
        target_norm = sigma_gradient_r
      elif self.norm_supervision_type == 'canonical':
        raise NotImplementedError
        # target_norm = sigma_gradient_w
      elif self.norm_supervision_type == 'direct':
        target_norm = sigma_gradient
      else:
        raise NotImplementedError
      target_norm = jnp.reshape(target_norm, (-1, num_samples, norm.shape[-1]))
      out['target_norm'] = target_norm

      # calculate back facing vectors
      pt_viewdirs = jnp.tile(jnp.expand_dims(viewdirs, 1), (1, num_samples, 1))
      back_facing = jnp.einsum('ijk,ijk->ij', norm, pt_viewdirs)
      back_facing = jnp.square(nn.relu(back_facing))
      out['back_facing'] = back_facing

    # accumulate sigma gradient for each ray
    weights = out['weights']
    # ray_sigma_gradient = (weights[..., None] * sigma_gradient_w).sum(axis=-2)
    # ray_sigma_gradient = (weights[..., None] * sigma_gradient).sum(axis=-2)
    if norm is not None:
      ray_sigma_gradient = (weights[..., None] * norm).sum(axis=-2)
    else:
      ray_sigma_gradient = jnp.zeros_like((weights[..., None] * sigma_gradient).sum(axis=-2))
    out['ray_sigma_gradient'] = ray_sigma_gradient
    # ray_sigma_gradient_r = (weights[..., None] * sigma_gradient_w).sum(axis=-2)
    ray_sigma_gradient_r = (weights[..., None] * sigma_gradient_r).sum(axis=-2)
    # ray_sigma_gradient_r = (weights[..., None] * warped_dummy_points).sum(axis=-2)
    out['ray_sigma_gradient_r'] = ray_sigma_gradient_r
    ray_rotation_field = (weights[..., None] * rotation_field).sum(axis=-2)
    out['ray_rotation_field'] = ray_rotation_field
    ray_translation_field = (weights[..., None] * translation_field).sum(axis=-2)
    out['ray_translation_field'] = ray_translation_field

    # accumulate hyper coordinates for each ray
    hyper_points = warped_points[..., 3:]
    ray_hyper_points = (weights[..., None] * hyper_points).sum(axis=-2)
    out['ray_hyper_points'] = ray_hyper_points

    # accumulate hyper c coordinates for each ray
    if self.use_hyper_c:
      hyper_c_shape = list(hyper_points.shape)
      hyper_c_shape[-1] = self.hyper_c_num_dims
      hyper_c = jnp.reshape(hyper_c, hyper_c_shape)
      ray_hyper_c = (weights[..., None] * hyper_c).sum(axis=-2)
      out['hyper_c'] = hyper_c
      out['ray_hyper_c'] = ray_hyper_c
      if hyper_c_jacobian is not None:
        out['hyper_c_jacobian'] = hyper_c_jacobian
    else:
      out['ray_hyper_c'] = jnp.zeros_like(ray_hyper_points)

    # accumulate inter norm for each ray
    if self.use_norm_voxel:
      if return_nv_details:
        out['inter_norm'] = inter_norm
        out['nv_vertex_values'] = nv_vertex_values
        out['nv_vertex_coef'] = nv_vertex_coef
      ray_inter_norm = (weights[..., None] * inter_norm).sum(axis=-2)
      out['ray_inter_norm'] = ray_inter_norm

    if self.mix_d_s_color:
      s_weight = s_weight.reshape([-1, num_samples, 1])
      ray_s_weights = (weights[..., None] * s_weight).sum(axis=-2)
      out['ray_s_weights'] = ray_s_weights
      d_color = d_color.reshape([-1, num_samples, 3])
      ray_d_color = (weights[..., None] * d_color).sum(axis=-2)
      out['ray_d_color'] = nn.sigmoid(ray_d_color)
      s_color = s_color.reshape([-1, num_samples, 3])
      ray_s_color = (weights[..., None] * s_color).sum(axis=-2)
      out['ray_s_color'] = nn.sigmoid(ray_s_color)

    if self.use_warp_c:
      color_delta_x = color_delta_x.reshape([-1, num_samples, 3])
      ray_color_delta_x = (weights[..., None] * color_delta_x).sum(axis=-2)
      out['ray_color_delta_x'] = ray_color_delta_x

    # hyper jacobian for regularization
    if hyper_jacobian is not None:
      out['hyper_jacobian'] = hyper_jacobian

    # Add a map containing the returned points at the median depth.
    depth_indices = model_utils.compute_depth_index(out['weights'])
    med_points = jnp.take_along_axis(
      # Unsqueeze axes: sample axis, coords.
      warped_points, depth_indices[..., None, None], axis=-2)
    out['med_points'] = med_points

    return out

  def __call__(
      self,
      rays_dict: Dict[str, Any],
      extra_params: Dict[str, Any],
      metadata_encoded=False,
      use_warp=True,
      return_points=False,
      return_weights=False,
      return_warp_jacobian=False,
      return_hyper_jacobian=False,
      return_hyper_c_jacobian=False,
      return_nv_details=True,
      near=None,
      far=None,
      use_sample_at_infinity=None,
      render_opts=None,
      deterministic=False,
      screw_input_mode=None,
      use_sigma_gradient=False,
      use_predicted_norm=False,
      norm_voxel_lr=0,
      norm_voxel_ratio=1,
  ):
    """Nerf Model.

    Args:
      rays_dict: a dictionary containing the ray information. Contains:
        'origins': the ray origins.
        'directions': unit vectors which are the ray directions.
        'viewdirs': (optional) unit vectors which are viewing directions.
        'metadata': a dictionary of metadata indices e.g., for warping.
      extra_params: parameters for the warp e.g., alpha.
      metadata_encoded: if True, assume the metadata is already encoded.
      use_warp: if True use the warp field (if also enabled in the model).
      return_points: if True return the points (and warped points if
        applicable).
      return_weights: if True return the density weights.
      return_warp_jacobian: if True computes and returns the warp Jacobians.
      near: if not None override the default near value.
      far: if not None override the default far value.
      use_sample_at_infinity: override for `self.use_sample_at_infinity`.
      render_opts: an optional dictionary of render options.
      deterministic: whether evaluation should be deterministic.

    Returns:
      ret: list, [(rgb, disp, acc), (rgb_coarse, disp_coarse, acc_coarse)]
    """
    use_warp = self.use_warp and use_warp
    # Extract viewdirs from the ray array
    origins = rays_dict['origins']
    directions = rays_dict['directions']
    metadata = rays_dict['metadata']
    if 'viewdirs' in rays_dict:
      viewdirs = rays_dict['viewdirs']
    else:  # viewdirs are normalized rays_d
      viewdirs = directions

    if near is None:
      near = self.near
    if far is None:
      far = self.far
    if use_sample_at_infinity is None:
      use_sample_at_infinity = self.use_sample_at_infinity

    # Evaluate coarse samples.
    z_vals, points = model_utils.sample_along_rays(
      self.make_rng('coarse'), origins, directions, self.num_coarse_samples,
      near, far, self.use_stratified_sampling,
      self.use_linear_disparity)

    coarse_ret = self.render_samples(
      'coarse',
      points,
      z_vals,
      directions,
      viewdirs,
      metadata,
      extra_params,
      use_warp=use_warp,
      metadata_encoded=metadata_encoded,
      return_warp_jacobian=return_warp_jacobian,
      return_hyper_jacobian=return_hyper_jacobian,
      return_hyper_c_jacobian=return_hyper_c_jacobian,
      return_nv_details=return_nv_details,
      use_sample_at_infinity=self.use_sample_at_infinity,
      screw_input_mode=screw_input_mode,
      use_sigma_gradient=use_sigma_gradient,
      use_predicted_norm=use_predicted_norm,
      norm_voxel_lr=norm_voxel_lr,
      norm_voxel_ratio=norm_voxel_ratio,
    )
    out = {'coarse': coarse_ret}

    # Evaluate fine samples.
    if self.num_fine_samples > 0:
      z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
      z_vals, points = model_utils.sample_pdf(
        self.make_rng('fine'), z_vals_mid, coarse_ret['weights'][..., 1:-1],
        origins, directions, z_vals, self.num_fine_samples,
        self.use_stratified_sampling)
      out['fine'] = self.render_samples(
        'fine',
        points,
        z_vals,
        directions,
        viewdirs,
        metadata,
        extra_params,
        use_warp=use_warp,
        metadata_encoded=metadata_encoded,
        return_warp_jacobian=return_warp_jacobian,
        return_hyper_jacobian=return_hyper_jacobian,
        return_hyper_c_jacobian=return_hyper_c_jacobian,
        return_nv_details=return_nv_details,
        use_sample_at_infinity=use_sample_at_infinity,
        render_opts=render_opts,
        screw_input_mode=screw_input_mode,
        use_sigma_gradient=use_sigma_gradient,
        use_predicted_norm=use_predicted_norm,
        norm_voxel_lr=norm_voxel_lr,
        norm_voxel_ratio=norm_voxel_ratio,
      )

    if not return_weights:
      del out['coarse']['weights']
      del out['fine']['weights']

    if not return_points:
      del out['coarse']['points']
      del out['coarse']['warped_points']
      del out['fine']['points']
      del out['fine']['warped_points']

    return out