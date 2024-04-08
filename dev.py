import lovely_jax as lj
import jax
from glcm_jax import glcm
from jax import config
config.update("jax_enable_x64", True)

lj.monkey_patch()
x0 = jax.random.uniform(jax.random.PRNGKey(0), (256,100), dtype="float32")
x1 = jax.random.uniform(jax.random.PRNGKey(1), (256,100), dtype="float32")
ecc = 0.1

out, = glcm(x0, x1)
print(out,)
print(jax.make_jaxpr(glcm)(x0, x1))
print(x0[0, :10])
print(x1[0, :10])
print(out[0, :10])

