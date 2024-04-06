import lovely_jax as lj
import jax
from kepler_jax import kepler

lj.monkey_patch()
mean_anom = jax.random.uniform(jax.random.PRNGKey(0), (256,100), dtype="float32")
ecc = 0.1

sinE, cosE = kepler(mean_anom, ecc)
print(sinE, cosE)
print(jax.make_jaxpr(kepler)(mean_anom, ecc))
