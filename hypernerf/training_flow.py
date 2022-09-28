import functools
from typing import Any, Callable, Dict

from absl import logging
import flax
from flax import struct
from flax import traverse_util
from flax.training import checkpoints
import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax import vmap

from hypernerf import model_utils
from hypernerf import models
from hypernerf import utils


@struct.dataclass
class ScalarParams:
  """Scalar parameters for flow model training"""
  learning_rate: float
  time_offset: float
  elastic_loss_weight: float = 0


def save_checkpoint(path, flow_only_path, state, keep=5):
  """Save the state to a checkpoint."""
  state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
  step = state_to_save.optimizer.state.step
  checkpoint_path = checkpoints.save_checkpoint(
      path, state_to_save, step, keep=keep)
  logging.info('Saved checkpoint: step=%d, path=%s', int(step), checkpoint_path)

  # save flow field params separately
  params = state_to_save.optimizer.target
  checkpoint_flow_only_path = checkpoints.save_checkpoint(
    flow_only_path, params, step, keep=keep
  )
  logging.info('Saved flow only checkpoint: step=%d, path=%s', int(step), checkpoint_flow_only_path)

  return checkpoint_path, checkpoint_flow_only_path


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
    svals = jnp.linalg.svd(jacobian + eps, compute_uv=False)
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
    raise NotImplementedError
    # rot = nearest_rotation_svd(jacobian)
    # sq_residual = jnp.sum((jacobian - rot) ** 2)
  else:
    raise NotImplementedError(
        f'Unknown elastic loss type {loss_type!r}')
  residual = jnp.sqrt(sq_residual)
  loss = utils.general_loss_with_squared_residual(
      sq_residual, alpha=-2.0, scale=0.03)
  return loss, residual


@functools.partial(jax.jit,
                   static_argnums=0,)
def train_step(
        model: models.FlowModel,
        rng_key: Callable[[int], jnp.ndarray],
        state: model_utils.TrainState,
        batch: Dict[str, Any],
        scalar_params: ScalarParams,
        nerf_params: Dict[str, Any],
):
  rng_key, fine_key, coarse_key = random.split(rng_key, 3)
  def _loss_fn(flow_params, nerf_params, scalar_params: ScalarParams):
    stats = {}
    params = {}
    params = flow_params['model']
    params['nerf_model'] = nerf_params['model']
    ret = model.apply(
      {'params': params},
      batch,
      extra_params=state.extra_params,
      time_offset=scalar_params.time_offset,
      rngs={
        'fine': fine_key,
        'coarse': coarse_key
      }
    )

    cur_sigma = ret['cur_sigma']
    warped_sigma = ret['warped_sigma']
    joint_weights = lax.stop_gradient(ret['joint_weights'])
    loss = (joint_weights * jnp.abs(cur_sigma - warped_sigma)).sum(-1).mean()

    stats['loss/sigma'] = loss

    stats['stats/sigma_80'] = jnp.percentile(cur_sigma, 80)
    stats['stats/sigma_50'] = jnp.percentile(cur_sigma, 50)
    stats['stats/sigma_20'] = jnp.percentile(cur_sigma, 20)

    delta_x = ret['delta_x']
    stats['stats/delta_x'] = jnp.percentile(jnp.abs(delta_x), 95)

    ray_delta_x = ret['ray_delta_x']
    stats['stats/ray_delta_x'] = jnp.percentile(jnp.abs(ray_delta_x), 95)

    # elastic_fn = functools.partial(compute_elastic_loss, loss_type='jtj')
    # v_elastic_fn = jax.jit(vmap(vmap(jax.jit(elastic_fn))))
    # jacobian = ret['warp_jacobian']
    #
    # elastic_loss, elastic_residual = v_elastic_fn(jacobian)
    # elastic_loss = weights * elastic_loss
    # elastic_loss = elastic_loss.sum(axis=-1).mean()

    elastic_loss = 0
    stats['loss/elastic'] = elastic_loss
    loss += scalar_params.elastic_loss_weight * elastic_loss

    stats['loss/total'] = loss

    return loss, (ret, stats)

  optimizer = state.optimizer

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, argnums=0)
  flow_params = optimizer.target
  (_, (model_out, stats)), grad = grad_fn(flow_params, nerf_params, scalar_params)

  grad = jax.lax.pmean(grad, axis_name='batch')
  stats = jax.lax.pmean(stats, axis_name='batch')
  model_out = jax.lax.pmean(model_out, axis_name='batch')

  new_optimizer = optimizer.apply_gradient(
      grad, learning_rate=scalar_params.learning_rate)
  new_state = state.replace(optimizer=new_optimizer)
  return new_state, stats, rng_key, model_out