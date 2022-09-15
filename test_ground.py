import jax
import jax.numpy as jnp
from hypernerf.model_utils import normalize_vector

def cal_ref_radiance(d, n):
  d = normalize_vector(d)
  n = normalize_vector(n)
  out = 2 * jnp.sum(d * n, axis=-1, keepdims=False)[..., jnp.newaxis] * n - d
  return out

d = jnp.array([1, 0, 0])
n = jnp.array([0, 1, 0])
d = jnp.tile(d[jnp.newaxis, jnp.newaxis, :], [10, 5, 1])
n = jnp.tile(n[jnp.newaxis, jnp.newaxis, :], [10, 5, 1])
print(cal_ref_radiance(d, n))
print(cal_ref_radiance(d, n).shape)
