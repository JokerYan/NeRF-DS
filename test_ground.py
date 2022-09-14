import jax
import jax.numpy as jnp
from hypernerf.model_utils import normalize_vector

def calculate_ref_radiance(d, n):
  d = normalize_vector(d)
  n = normalize_vector(n)
  out = 2 * d.transpose() @ n * n - d
  return out

d = jnp.array([1, 0, 0])
n = jnp.array([0, 1, 0])
d = jnp.tile(d[jnp.newaxis, :], [10, 1])
n = jnp.tile(n[jnp.newaxis, :], [10, 1])
print(calculate_ref_radiance(d, n))
