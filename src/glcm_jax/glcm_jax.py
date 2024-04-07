# -*- coding: utf-8 -*-

__all__ = ["glcm"]

from functools import partial

import numpy as np
from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

# Register the CPU XLA custom calls
from . import cpu_ops

for _name, _value in cpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")

# If the GPU version exists, also register those
try:
    from . import gpu_ops
except ImportError:
    gpu_ops = None
else:
    for _name, _value in gpu_ops.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

# This function exposes the primitive to user code and this is the only
# public-facing function in this module

def glcm(x0, x1):
    # We're going to apply array broadcasting here since the logic of our op
    # is much simpler if we require the inputs to all have the same shapes
    x0, x1 = jnp.broadcast_arrays(x0, x1)
    return _glcm_prim.bind(x0, x1)

# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _glcm_abstract(x0, x1):
    shape = x0.shape
    dtype = dtypes.canonicalize_dtype(x0.dtype)
    assert dtypes.canonicalize_dtype(x1.dtype) == dtype
    assert x1.shape == shape
    return (ShapedArray(shape, dtype), ShapedArray(shape, dtype))


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
# This provides a mechanism for exposing our custom C++ and/or CUDA interfaces
# to the JAX XLA backend. We're wrapping two translation rules into one here:
# one for the CPU and one for the GPU
def _glcm_lowering(ctx, x0, x1, *, platform="cpu"):

    # Checking that input types and shape agree
    assert x0.type == x1.type

    # Extract the numpy type of the inputs
    x0_aval, _ = ctx.avals_in
    np_dtype = np.dtype(x0_aval.dtype)

    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(x0.type)
    dims = dtype.shape
    layout = tuple(range(len(dims) - 1, -1, -1))

    # The total size of the input is the product across dimensions
    size = np.prod(dims).astype(np.int64)

    # We dispatch a different call depending on the dtype
    if np_dtype == np.float32:
        op_name = platform + "_addition_f32"
    elif np_dtype == np.float64:
        op_name = platform + "_addition_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we pass the size of the data as a the first input
        # argument
        return custom_call(
            op_name,
            # Output types
            result_types=[dtype, ],
            # The inputs:
            operands=[mlir.ir_constant(size), x0, x1],
            # Layout specification:
            operand_layouts=[(), layout, layout],
            result_layouts=[layout, ]
        ).results

    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'glcm_jax' module was not compiled with CUDA support"
            )
        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        opaque = gpu_ops.build_glcm_descriptor(size)

        return custom_call(
            op_name,
            # Output types
            result_types=[dtype, ],
            # The inputs:
            operands=[x0, x1],
            # Layout specification:
            operand_layouts=[layout, layout],
            result_layouts=[layout, ],
            # GPU specific additional data
            backend_config=opaque
        ).results

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu'"
    )


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************

# Here we define the differentiation rules using a JVP derived using finite differences
# of the GLCM:
#
# TODO: this might need to be a different CUDA kernel...
# In this case we don't need to define a transpose rule in order to support
# reverse and higher order differentiation. This might not be true in other
# applications, so check out the "How JAX primitives work" tutorial in the JAX
# documentation for more info as necessary.
def _glcm_jvp(args, tangents):
    x0, x1 = args
    d_x0, d_x1 = tangents

    # We use "bind" here because we don't want to mod the mean anomaly again
    out, = _glcm_prim.bind(x0, x1)

    def zero_tangent(tan, val):
        return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan

    return (out,), (zero_tangent(jnp.array(0.0), out),)


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************

# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _glcm_batch(args, axes):
    assert axes[0] == axes[1]
    return glcm(*args), axes


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_glcm_prim = core.Primitive("glcm")
_glcm_prim.multiple_results = True
_glcm_prim.def_impl(partial(xla.apply_primitive, _glcm_prim))
_glcm_prim.def_abstract_eval(_glcm_abstract)

# Connect the XLA translation rules for JIT compilation
for platform in ["cpu", "gpu"]:
    mlir.register_lowering(
        _glcm_prim,
        partial(_glcm_lowering, platform=platform),
        platform=platform)

# Connect the JVP and batching rules
ad.primitive_jvps[_glcm_prim] = _glcm_jvp
batching.primitive_batchers[_glcm_prim] = _glcm_batch
