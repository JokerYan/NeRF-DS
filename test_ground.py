import jax
import jax.numpy as jnp
from hypernerf.model_utils import normalize_vector

def calculate_ref_radiance(d, n):
  d = normalize_vector(d)
  n = normalize_vector(n)
