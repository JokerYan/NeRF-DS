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

"""Library to training NeRFs."""
import functools
from typing import Any, Callable, Dict

from absl import logging
import flax
from flax import struct
from flax import traverse_util
from flax.core import FrozenDict
from flax.training import checkpoints
import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax import vmap
from optax._src.loss import sigmoid_binary_cross_entropy

from hypernerf import model_utils
from hypernerf import models
from hypernerf import utils
from hypernerf import camera as cam
from hypernerf.utils import grid_sample


@struct.dataclass
class ScalarParams:
  """Scalar parameters for training."""
  learning_rate: float
  elastic_loss_weight: float = 0.0
  warp_reg_loss_weight: float = 0.0
  warp_reg_loss_alpha: float = -2.0
  warp_reg_loss_scale: float = 0.001
  background_loss_weight: float = 0.0
  background_noise_std: float = 0.001
  hyper_reg_loss_weight: float = 0.0
  sigma_grad_diff_reg_weight: float = 0.0
  back_facing_reg_weight: float = 0.0
  hyper_concentration_reg_weight: float = 0.0
  hyper_concentration_reg_scale: float = 0.0
  hyper_jacobian_reg_weight: float = 0.0
  hyper_jacobian_reg_scale: float = 0.0
  hyper_c_jacobian_reg_weight: float = 0.0
  hyper_c_jacobian_reg_scale: float = 0.0
  norm_voxel_loss_weight: float = 0.0
  flow_model_light_learning_rate: float = 0.0
  mask_weight: float = 0.5
  in_mask_consistency_loss_weight: float = 1.0
  out_mask_consistency_loss_weight: float = 1.0
  predicted_mask_loss_weight: float = 1.0
  mask_ratio: float = 1.0


def save_checkpoint(path, state, keep=2):
  """Save the state to a checkpoint."""
  state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
  step = state_to_save.optimizer.state.step
  checkpoint_path = checkpoints.save_checkpoint(
      path, state_to_save, step, keep=keep)
  logging.info('Saved checkpoint: step=%d, path=%s', int(step), checkpoint_path)
  return checkpoint_path


def zero_adam_param_states(state: flax.optim.OptimizerState, selector: str):
  """Applies a gradient for a set of parameters.

  Args:
    state: a named tuple containing the state of the optimizer
    selector: a path string defining which parameters to freeze.

  Returns:
    A tuple containing the new parameters and the new optimizer state.
  """
  step = state.step
  params = flax.core.unfreeze(state.param_states)
  flat_params = {'/'.join(k): v
                 for k, v in traverse_util.flatten_dict(params).items()}
  for k in flat_params:
    if k.startswith(selector):
      v = flat_params[k]
      # pylint: disable=protected-access
      flat_params[k] = flax.optim.adam._AdamParamState(
          jnp.zeros_like(v.grad_ema), jnp.zeros_like(v.grad_sq_ema))

  new_param_states = traverse_util.unflatten_dict(
      {tuple(k.split('/')): v for k, v in flat_params.items()})
  new_param_states = dict(flax.core.freeze(new_param_states))
  new_state = flax.optim.OptimizerState(step, new_param_states)
  return new_state


@jax.jit
def nearest_rotation_svd(matrix, eps=1e-6):
  """Computes the nearest rotation using SVD."""
  # TODO(keunhong): Currently this produces NaNs for some reason.
  u, _, vh = jnp.linalg.svd(matrix + eps, compute_uv=True, full_matrices=False)
  # Handle the case when there is a flip.
  # M will be the identity matrix except when det(UV^T) = -1
  # in which case the last diagonal of M will be -1.
  det = jnp.linalg.det(utils.matmul(u, vh))
  m = jnp.stack([jnp.ones_like(det), jnp.ones_like(det), det], axis=-1)
  m = jnp.diag(m)
  r = utils.matmul(u, utils.matmul(m, vh))
  return r


def compute_elastic_loss(jacobian, eps=1e-6, loss_type='log_svals'):
  """Compute the elastic regularization loss.

  The loss is given by sum(log(S)^2). This penalizes the singular values
  when they deviate from the identity since log(1) = 0.0,
  where D is the diagonal matrix containing the singular values.

  Args:
    jacobian: the Jacobian of the point transformation.
    eps: a small value to prevent taking the log of zero.
    loss_type: which elastic loss type to use.

  Returns:
    The elastic regularization loss.
  """
  if loss_type == 'log_svals':
    svals = jnp.linalg.svd(jacobian, compute_uv=False)
    log_svals = jnp.log(jnp.maximum(svals, eps))
    sq_residual = jnp.sum(log_svals**2, axis=-1)
  elif loss_type == 'svals':
    svals = jnp.linalg.svd(jacobian, compute_uv=False)
    sq_residual = jnp.sum((svals - 1.0)**2, axis=-1)
  elif loss_type == 'jtj':
    jtj = utils.matmul(jacobian, jacobian.T)
    sq_residual = ((jtj - jnp.eye(3)) ** 2).sum() / 4.0
  elif loss_type == 'div':
    div = utils.jacobian_to_div(jacobian)
    sq_residual = div ** 2
  elif loss_type == 'det':
    det = jnp.linalg.det(jacobian)
    sq_residual = (det - 1.0) ** 2
  elif loss_type == 'log_det':
    det = jnp.linalg.det(jacobian)
    sq_residual = jnp.log(jnp.maximum(det, eps)) ** 2
  elif loss_type == 'nr':
    rot = nearest_rotation_svd(jacobian)
    sq_residual = jnp.sum((jacobian - rot) ** 2)
  else:
    raise NotImplementedError(
        f'Unknown elastic loss type {loss_type!r}')
  residual = jnp.sqrt(sq_residual)
  loss = utils.general_loss_with_squared_residual(
      sq_residual, alpha=-2.0, scale=0.03)
  return loss, residual


@functools.partial(jax.jit, static_argnums=0)
def compute_background_loss(model, state, params, key, points, noise_std,
                            alpha=-2, scale=0.001):
  """Compute the background regularization loss."""
  metadata = random.choice(key, model.warp_embeds, shape=(points.shape[0], 1))
  point_noise = noise_std * random.normal(key, points.shape)
  points = points + point_noise
  warp_fn = functools.partial(model.apply, method=model.apply_warp)
  warp_fn = jax.vmap(warp_fn, in_axes=(None, 0, 0, None))
  warp_out = warp_fn({'params': params}, points, metadata, state.extra_params)
  warped_points = warp_out['warped_points'][..., :3]
  sq_residual = jnp.sum((warped_points - points)**2, axis=-1)
  loss = utils.general_loss_with_squared_residual(
      sq_residual, alpha=alpha, scale=scale)
  return loss


@functools.partial(jax.jit,
                   static_argnums=0,
                   static_argnames=('disable_hyper_grads',
                                    'grad_max_val',
                                    'grad_max_norm',
                                    'use_elastic_loss',
                                    'elastic_reduce_method',
                                    'elastic_loss_type',
                                    'use_background_loss',
                                    'use_warp_reg_loss',
                                    'use_hyper_reg_loss',
                                    'screw_input_mode',
                                    'use_sigma_gradient',
                                    'use_sigma_grad_diff_reg',
                                    'use_predicted_norm',
                                    'use_back_facing_reg',
                                    'use_hyper_concentration_reg',
                                    'use_hyper_jacobian_reg',
                                    'use_hyper_c_jacobian_reg',
                                    'use_mask_weighted_loss',
                                    'use_mask_consistency_loss',
                                    'canonical_camera',
                                    ))
def train_step(model: models.CustomModel,
               rng_key: Callable[[int], jnp.ndarray],
               state: model_utils.TrainState,
               batch: Dict[str, Any],
               scalar_params: ScalarParams,
               disable_hyper_grads: bool = False,
               grad_max_val: float = 0.0,
               grad_max_norm: float = 0.0,
               use_elastic_loss: bool = False,
               elastic_reduce_method: str = 'median',
               elastic_loss_type: str = 'log_svals',
               use_background_loss: bool = False,
               use_warp_reg_loss: bool = False,
               use_hyper_reg_loss: bool = False,
               screw_input_mode: str = None,
               use_sigma_gradient: bool = False,
               use_sigma_grad_diff_reg: bool = False,
               use_predicted_norm: bool = False,
               use_back_facing_reg: bool = False,
               use_hyper_concentration_reg: bool = False,
               use_hyper_jacobian_reg: bool = False,
               use_hyper_c_jacobian_reg: bool = False,
               use_mask_weighted_loss: bool = False,
               use_mask_consistency_loss: bool = False,
               canonical_camera: cam.Camera = None,
               ):
  """One optimization step.

  Args:
    model: the model module to evaluate.
    rng_key: The random number generator.
    state: model_utils.TrainState, state of model and optimizer.
    batch: dict. A mini-batch of data for training.
    scalar_params: scalar-valued parameters.
    disable_hyper_grads: if True disable gradients to the hyper coordinate
      branches.
    grad_max_val: The gradient clipping value (disabled if == 0).
    grad_max_norm: The gradient clipping magnitude (disabled if == 0).
    use_elastic_loss: is True use the elastic regularization loss.
    elastic_reduce_method: which method to use to reduce the samples for the
      elastic loss. 'median' selects the median depth point sample while
      'weight' computes a weighted sum using the density weights.
    elastic_loss_type: which method to use for the elastic loss.
    use_background_loss: if True use the background regularization loss.
    use_warp_reg_loss: if True use the warp regularization loss.
    use_hyper_reg_loss: if True regularize the hyper points.
    screw_input_mode: whether screw axis is used in rgb rendering ["None", "rotation", "full"]

  Returns:
    new_state: model_utils.TrainState, new training state.
    stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
  """
  rng_key, fine_key, coarse_key, reg_key, voxel_key = random.split(rng_key, 5)

  # pylint: disable=unused-argument
  def _compute_loss_and_stats(
      params, model_out, level,
      use_elastic_loss=False,
      use_hyper_reg_loss=False):

    stats = {}
    if 'channel_set' in batch['metadata']:
      num_sets = int(model_out['rgb'].shape[-1] / 3)
      losses = []
      for i in range(num_sets):
        loss = (model_out['rgb'][..., i * 3:(i + 1) * 3] - batch['rgb'])**2
        loss *= (batch['metadata']['channel_set'] == i)
        losses.append(loss)
      rgb_loss = jnp.sum(jnp.asarray(losses), axis=0)
    else:
      rgb_loss = ((model_out['rgb'][..., :3] - batch['rgb'][..., :3])**2)

    # mask weighted loss
    if use_mask_weighted_loss:
      mask = batch['mask']
      mask_weight = scalar_params.mask_weight
      in_mask_rgb_loss = mask_weight * mask * rgb_loss
      out_mask_rgb_loss = (1 - mask_weight) * (1 - mask) * rgb_loss
      stats['loss/in_mask_rgb'] = in_mask_rgb_loss.mean()
      stats['loss/out_mask_rgb'] = out_mask_rgb_loss.mean()
      rgb_loss = in_mask_rgb_loss + out_mask_rgb_loss

    rgb_loss = rgb_loss.mean()

    stats['loss/rgb'] = rgb_loss
    loss = rgb_loss
    if use_elastic_loss:
      elastic_fn = functools.partial(compute_elastic_loss,
                                     loss_type=elastic_loss_type)
      v_elastic_fn = jax.jit(vmap(vmap(jax.jit(elastic_fn))))
      weights = lax.stop_gradient(model_out['weights'])
      jacobian = model_out['warp_jacobian']
      # Pick the median point Jacobian.
      if elastic_reduce_method == 'median':
        depth_indices = model_utils.compute_depth_index(weights)
        jacobian = jnp.take_along_axis(
            # Unsqueeze axes: sample axis, Jacobian row, Jacobian col.
            jacobian, depth_indices[..., None, None, None], axis=-3)
      # Compute loss using Jacobian.
      elastic_loss, elastic_residual = v_elastic_fn(jacobian)
      # Multiply weight if weighting by density.
      if elastic_reduce_method == 'weight':
        elastic_loss = weights * elastic_loss
      elastic_loss = elastic_loss.sum(axis=-1).mean()
      stats['loss/elastic'] = elastic_loss
      stats['residual/elastic'] = jnp.mean(elastic_residual)
      loss += scalar_params.elastic_loss_weight * elastic_loss

    if use_warp_reg_loss:
      weights = lax.stop_gradient(model_out['weights'])
      depth_indices = model_utils.compute_depth_index(weights)
      warp_mag = ((model_out['points']
                   - model_out['warped_points'][..., :3]) ** 2).sum(axis=-1)
      warp_reg_residual = jnp.take_along_axis(
          warp_mag, depth_indices[..., None], axis=-1)
      warp_reg_loss = utils.general_loss_with_squared_residual(
          warp_reg_residual,
          alpha=scalar_params.warp_reg_loss_alpha,
          scale=scalar_params.warp_reg_loss_scale).mean()
      stats['loss/warp_reg'] = warp_reg_loss
      stats['residual/warp_reg'] = jnp.mean(jnp.sqrt(warp_reg_residual))
      loss += scalar_params.warp_reg_loss_weight * warp_reg_loss

    if use_hyper_reg_loss:
      weights = lax.stop_gradient(model_out['weights'])
      hyper_points = model_out['warped_points'][..., 3:]
      hyper_reg_residual = (hyper_points ** 2).sum(axis=-1)
      hyper_reg_loss = utils.general_loss_with_squared_residual(
          hyper_reg_residual, alpha=0.0, scale=0.05)
      hyper_reg_loss = (weights * hyper_reg_loss).sum(axis=1).mean()
      stats['loss/hyper_reg'] = hyper_reg_loss
      stats['residual/hyper_reg'] = jnp.mean(jnp.sqrt(hyper_reg_residual))
      loss += scalar_params.hyper_reg_loss_weight * hyper_reg_loss

    if use_sigma_grad_diff_reg:
      sigma_grad_diff = jnp.mean(model_out['sigma_grad_diff'])
      stats['loss/sigma_grad_diff'] = sigma_grad_diff
      loss += scalar_params.sigma_grad_diff_reg_weight * sigma_grad_diff

    if use_predicted_norm:
      weights = lax.stop_gradient(model_out['weights'])
      predicted_norm = model_out['predicted_norm']
      target_norm = model_out['target_norm']
      # norm_diff = 1 - jnp.einsum('ijk,ijk->ij', predicted_norm, target_norm)
      norm_diff = jnp.linalg.norm(predicted_norm - target_norm, axis=-1, ord=2)
      norm_diff_loss = jnp.mean(weights * norm_diff)
      # norm_diff_loss = jnp.mean(model_out['norm_diff'])
      stats['loss/norm_diff_loss'] = norm_diff_loss
      loss += state.norm_loss_weight * norm_diff_loss

    if use_back_facing_reg:
      weights = lax.stop_gradient(model_out['weights'])
      back_facing = model_out['back_facing']
      back_facing = weights * back_facing
      back_facing_loss = jnp.mean(back_facing)
      stats['loss/back_facing_loss'] = back_facing_loss
      loss += scalar_params.back_facing_reg_weight * back_facing_loss

    if use_hyper_concentration_reg:
      weights = lax.stop_gradient(model_out['weights'])
      hyper_points = model_out['warped_points'][..., 3:]
      # hyper_concentration_reg_loss = utils.general_loss_with_squared_residual(
      #   hyper_points, alpha=-2, scale=scalar_params.hyper_concentration_reg_scale
      # )
      hyper_concentration_reg_loss = utils.gm_loss(
        hyper_points, scale=scalar_params.hyper_concentration_reg_scale
      )
      # assert weights.shape == 0, (weights.shape, hyper_concentration_reg_loss.shape)
      hyper_concentration_reg_loss = hyper_concentration_reg_loss.sum(axis=-1)                    # sum over the w coordinates
      hyper_concentration_reg_loss = (weights * hyper_concentration_reg_loss).sum(axis=1).mean()  # sum over the same ray, mean over different rays
      stats['loss/hyper_concentration_reg_loss'] = hyper_concentration_reg_loss
      loss += scalar_params.hyper_concentration_reg_weight * hyper_concentration_reg_loss

      hyper_coord_scale = jnp.abs(hyper_points)
      hyper_coord_scale = jnp.mean(hyper_coord_scale)
      stats['stats/hyper_coord_scale'] = hyper_coord_scale

      hyper_coord_top = jnp.percentile(jnp.abs(hyper_points).reshape(-1), 90)
      stats['stats/hyper_coord_top'] = hyper_coord_top

    if use_hyper_jacobian_reg:
      weights = lax.stop_gradient(model_out['weights'])
      hyper_jacobian = model_out['hyper_jacobian']
      hyper_jacobian = hyper_jacobian.reshape(hyper_jacobian.shape[0], hyper_jacobian.shape[1], -1)
      hyper_jacobian_reg_loss = utils.gm_loss(hyper_jacobian, scale=scalar_params.hyper_jacobian_reg_scale)
      hyper_jacobian_reg_loss = hyper_jacobian_reg_loss.sum(axis=-1)
      hyper_jacobian_reg_loss = (weights * hyper_jacobian_reg_loss).sum(axis=1).mean()
      stats['loss/hyper_jacobian_reg_loss'] = hyper_jacobian_reg_loss
      loss += scalar_params.hyper_jacobian_reg_weight * hyper_jacobian_reg_loss

      hyper_jacobian_scale = jnp.mean(jnp.abs(hyper_jacobian))
      stats['stats/hyper_jacobian_scale'] = hyper_jacobian_scale

    if use_hyper_c_jacobian_reg:
      weights = lax.stop_gradient(model_out['weights'])
      hyper_c_jacobian = model_out['hyper_c_jacobian']
      hyper_c_jacobian = hyper_c_jacobian.reshape(hyper_c_jacobian.shape[0], hyper_c_jacobian.shape[1], -1)
      hyper_c_jacobian_reg_loss = utils.gm_loss(hyper_c_jacobian, scale=scalar_params.hyper_c_jacobian_reg_scale)
      hyper_c_jacobian_reg_loss = hyper_c_jacobian_reg_loss.sum(axis=-1)
      hyper_c_jacobian_reg_loss = (weights * hyper_c_jacobian_reg_loss).sum(axis=1).mean()
      stats['loss/hyper_c_jacobian_reg_loss'] = hyper_c_jacobian_reg_loss
      loss += scalar_params.hyper_c_jacobian_reg_weight * hyper_c_jacobian_reg_loss

      hyper_c_jacobian_scale = jnp.mean(jnp.abs(hyper_c_jacobian))
      stats['stats/hyper_c_jacobian_scale'] = hyper_c_jacobian_scale

    if 'warp_jacobian' in model_out:
      jacobian = model_out['warp_jacobian']
      jacobian_det = jnp.linalg.det(jacobian)
      jacobian_div = utils.jacobian_to_div(jacobian)
      jacobian_curl = utils.jacobian_to_curl(jacobian)
      stats['metric/jacobian_det'] = jnp.mean(jacobian_det)
      stats['metric/jacobian_div'] = jnp.mean(jacobian_div)
      stats['metric/jacobian_curl'] = jnp.mean(
          jnp.linalg.norm(jacobian_curl, axis=-1))

    if 'sigma_grad_diff' in model_out:
      stats['stats/sigma_grad_diff'] = jnp.mean(model_out['sigma_grad_diff'])

    if 'inter_norm' in model_out:
      weights = lax.stop_gradient(model_out['weights'])   # R x S
      sigma = lax.stop_gradient(model_out['sigma'])   # R x S
      inv_sigma = 1 - jnp.exp(-sigma)   # R x S

      predicted_norm = lax.stop_gradient(model_out['predicted_norm'])   # R x S x 3
      inter_norm = model_out['inter_norm']
      nv_vertex_values = model_out['nv_vertex_values']    # R x S x 8 x 3
      nv_vertex_coef = model_out['nv_vertex_coef']        # R x S x 8

      # inter_norm_loss = jnp.mean(jnp.square(predicted_norm - inter_norm), axis=-1)

      predicted_norm_expanded = jnp.tile(jnp.expand_dims(predicted_norm, axis=2), [1, 1, 8, 1])   # R x S x 8 x 3
      inter_norm_loss = jnp.mean(jnp.square(predicted_norm_expanded - nv_vertex_values), axis=-1) # R x S x 8
      inter_norm_loss = jnp.sum(inter_norm_loss * nv_vertex_coef, axis=-1)  # scale by inter coef, R x S

      # inter_norm_loss = jnp.mean(weights * inter_norm_loss)
      inter_norm_loss = jnp.mean(inv_sigma * inter_norm_loss)

      stats['loss/inter_norm_loss'] = inter_norm_loss
      loss += scalar_params.norm_voxel_loss_weight * inter_norm_loss

    if use_mask_consistency_loss:
      canonical_mask = canonical_camera.get_mask().squeeze(axis=-1)
      cur_mask = batch['mask']
      warped_points = model_out['warped_points'][..., :3]

      # project
      warped_points_2d = canonical_camera.project_jnp(warped_points)
      warped_points_2d = jnp.concatenate([warped_points_2d[..., 1, jnp.newaxis],
                                          warped_points_2d[..., 0, jnp.newaxis]], axis=-1)    # x,y to y,x

      # grid sample
      warped_mask = grid_sample(canonical_mask, warped_points_2d)

      # in mask consistency loss
      weights = lax.stop_gradient(model_out['weights'])
      cur_mask = jnp.broadcast_to(cur_mask, warped_mask.shape)  # broadcast to samples along the ray
      in_mask_consistency_loss = (cur_mask * (1 - warped_mask) * weights).sum(axis=1).mean()
      stats['loss/in_mask_cons_loss'] = in_mask_consistency_loss
      loss += scalar_params.in_mask_consistency_loss_weight * in_mask_consistency_loss

      # out mask consistency loss
      delta_x = model_out['delta_x']
      delta_x_magnitude = jnp.linalg.norm(delta_x, axis=-1)
      # assert delta_x_magnitude.shape == 0, (delta_x_magnitude.shape, weights.shape)   # both 512 x 128
      out_mask_consistency_loss = ((1 - cur_mask) * delta_x_magnitude * weights).sum(axis=1).mean()
      stats['loss/out_mask_cons_loss'] = out_mask_consistency_loss
      loss += scalar_params.out_mask_consistency_loss_weight * out_mask_consistency_loss

      in_mask_delta_x = (cur_mask * delta_x_magnitude * weights).sum(axis=1).mean()
      stats['stats/in_mask_delta_x'] = in_mask_delta_x

    if 'predicted_mask' in model_out:
      alpha = lax.stop_gradient(model_out['alpha'])       # R x S   1 - exp(-sigma * dist)
      normalized_alpha = alpha / jnp.sum(alpha, axis=1)[:, jnp.newaxis]
      weights = lax.stop_gradient(model_out['weights'])   # R x S
      normalized_weights = weights / jnp.sum(weights, axis=1)[:, jnp.newaxis]

      predicted_mask = model_out['predicted_mask'].squeeze(axis=-1)   # R x S
      gt_mask = batch['mask']                                         # R x 1
      gt_mask = jnp.broadcast_to(gt_mask, predicted_mask.shape)       # R x S

      # # supervise 3d mask
      # mask_diff = jnp.abs(predicted_mask - weights * gt_mask)  # R x S
      # predicted_mask_loss = (normalized_alpha * mask_diff).sum(axis=1).mean()

      # supervise 2d mask
      mask_diff = jnp.abs(predicted_mask - gt_mask)
      # mask_diff = sigmoid_binary_cross_entropy(predicted_mask, gt_mask)
      predicted_mask_loss = (weights * mask_diff).sum(axis=1).mean()

      stats['loss/predicted_mask_loss'] = predicted_mask_loss
      loss += scalar_params.predicted_mask_loss_weight * predicted_mask_loss

      stats['stats/predicted_mask_max'] = jnp.max(predicted_mask)
      stats['stats/predicted_mask_sigmoid_max'] = jnp.max(jax.nn.sigmoid(predicted_mask))
      stats['stats/gt_mask_max'] = jnp.max(gt_mask)

    stats['loss/total'] = loss
    stats['metric/psnr'] = utils.compute_psnr(rgb_loss)

    # test points range
    points = model_out['points']
    min_x, max_x = jnp.min(points), jnp.max(points)
    stats['stats/min_x'] = min_x
    stats['stats/max_x'] = max_x

    return loss, stats

  def _loss_fn(params):
    ret = model.apply({'params': params['model']},
                      batch,
                      extra_params=state.extra_params,
                      return_points=True,
                      return_weights=True,
                      return_warp_jacobian=use_elastic_loss,
                      return_hyper_jacobian=use_hyper_jacobian_reg,
                      return_hyper_c_jacobian=use_hyper_c_jacobian_reg,
                      rngs={
                          'voxel': voxel_key,
                          'fine': fine_key,
                          'coarse': coarse_key
                      },
                      screw_input_mode=screw_input_mode,
                      use_sigma_gradient=use_sigma_gradient,
                      use_predicted_norm=use_predicted_norm,
                      norm_voxel_lr=state.norm_voxel_lr,
                      norm_voxel_ratio=state.norm_voxel_ratio,
                      mask_ratio=scalar_params.mask_ratio
                      )

    losses = {}
    stats = {}
    if 'fine' in ret:
      losses['fine'], stats['fine'] = _compute_loss_and_stats(
          params, ret['fine'], 'fine')
    if 'coarse' in ret:
      losses['coarse'], stats['coarse'] = _compute_loss_and_stats(
          params, ret['coarse'], 'coarse',
          use_elastic_loss=use_elastic_loss,
          use_hyper_reg_loss=use_hyper_reg_loss)

    if use_background_loss:
      background_loss = compute_background_loss(
          model,
          state=state,
          params=params['model'],
          key=reg_key,
          points=batch['background_points'],
          noise_std=scalar_params.background_noise_std)
      background_loss = background_loss.mean()
      losses['background'] = (
          scalar_params.background_loss_weight * background_loss)
      stats['loss/background_loss'] = background_loss

    return sum(losses.values()), (stats, ret)

  optimizer = state.optimizer

  if disable_hyper_grads:
    optimizer = optimizer.replace(
        state=zero_adam_param_states(optimizer.state, 'model/hyper_sheet_mlp'))
  # # disable flow model optimization
  ## Does not seem to freeze the parameters
  # optimizer = optimizer.replace(
  #   state=zero_adam_param_states(optimizer.state, 'model/flow_model')
  # )

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (_, (stats, model_out)), grad = grad_fn(optimizer.target)

  # # remove flow model from grad
  # model_grad = grad['model']
  # model_grad = model_grad.pop('flow_model')
  # grad['model'] = model_grad

  grad = jax.lax.pmean(grad, axis_name='batch')
  if grad_max_val > 0.0 or grad_max_norm > 0.0:
    grad = utils.clip_gradients(grad, grad_max_val, grad_max_norm)
  stats = jax.lax.pmean(stats, axis_name='batch')
  model_out = jax.lax.pmean(model_out, axis_name='batch')

  # # DEBUG grad for norm voxel
  # norm_voxel_grad = lax.stop_gradient(grad['model']['norm_voxel']['norm_voxel_array'])
  # norm_voxel_grad = jnp.max(jnp.abs(norm_voxel_grad))
  # stats['norm_voxel_grad'] = norm_voxel_grad

  if model.use_flow_model:
    hparams = optimizer.optimizer_def.hyper_params
    new_optimizer = optimizer.apply_gradient(
      grad,
      hyper_params=[
        hparams[0].replace(learning_rate=scalar_params.learning_rate),
        hparams[1].replace(learning_rate=scalar_params.flow_model_light_learning_rate)
      ]
    )
  else:
    new_optimizer = optimizer.apply_gradient(
        grad, learning_rate=scalar_params.learning_rate)
  new_state = state.replace(optimizer=new_optimizer)
  return new_state, stats, rng_key, model_out
