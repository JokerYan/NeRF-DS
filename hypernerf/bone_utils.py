import time
import jax
import jax.numpy as jnp

from hypernerf.quaternion import to_rotation_matrix


def get_bone_probs(points, bone_centers, bone_scales, bone_quaternions):
  """
  Calculate the probability of a points belonging to the bones based on Mahalanobis distance.
  Inputs:
    points: N x 3
    bone_centers: B x 3
    bone_scales: B x 3
    bone_quaternions: B x 4
  Outputs:
    bone_probs: N x B
  """
  N, B = points.shape[0], bone_centers.shape[0]
  points = jnp.broadcast_to(points[:, jnp.newaxis, :], [N, B, 3])
  bone_centers = jnp.broadcast_to(bone_centers[jnp.newaxis, :, :], [N, B, 3])
  delta_p = points - bone_centers     # N x B x 3

  # rotate
  bone_rotations = to_rotation_matrix(bone_quaternions)   # B x 3 x 3
  bone_rotations = jnp.broadcast_to(bone_rotations[jnp.newaxis, ...], [N, B, 3, 3])   # N x B x 3 x 3
  delta_p = (bone_rotations.transpose(0, 1, 3, 2) @ delta_p[..., jnp.newaxis]).squeeze(-1)          # N x B x 3

  # Mahalanobis distance
  bone_scales = jnp.broadcast_to(bone_scales[jnp.newaxis, ...], [N, B, 3])
  m_dist_square = jnp.square(delta_p) * (1 / bone_scales)       # N x B x 3
  m_dist_square = jnp.sum(m_dist_square, axis=-1)               # N x B

  # prob
  normalizer = 1 / jnp.sqrt(2 * jnp.pi * jnp.product(bone_scales, axis=-1))     # N * B
  prob = normalizer * jnp.exp(- m_dist_square / 2)

  return prob


if __name__ == "__main__":
  points = jnp.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])
  bone_centers = jnp.array([[0, 0, 0],
                            [0, 0, 0]])
  bone_scales = jnp.array([[1, 1, 1],
                           [1, 2, 1]])
  bone_quaternions = jnp.array([[0, 0, 0, 1],
                                [0, 0, 0, 1]])
  bone_probs = get_bone_probs(points, bone_centers, bone_scales, bone_quaternions)
  print(bone_probs)

  # start_time = time.time()
  # N = 5000
  # B = 5
  # key = jax.random.PRNGKey(0)
  # points = jax.random.normal(key, (N, 3))
  # bone_centers = jax.random.normal(key, (B, 3))
  # bone_scales = jax.random.normal(key, (B, 3))
  # bone_quaternions = jax.random.normal(key, (B, 4))
  # bone_probs = get_bone_probs(points, bone_centers, bone_scales, bone_quaternions)
  # print(time.time() - start_time)
