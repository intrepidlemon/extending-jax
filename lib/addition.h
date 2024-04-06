// This header defines the actual algorithm for our op. It is reused in cpu_ops.cc and
// kernels.cc.cu to expose this as a XLA custom call. The details aren't too important
// except that directly implementing this algorithm as a higher-level JAX function
// probably wouldn't be very efficient. That being said, this is not meant as a
// particularly efficient or robust implementation. It's just here to demonstrate the
// infrastructure required to extend JAX.

#ifndef _GLCM_JAX_KEPLER_H_
#define _GLCM_JAX_KEPLER_H_

#include <cmath>

namespace glcm_jax {

#ifdef __CUDACC__
#define GLCM_JAX_INLINE_OR_DEVICE __host__ __device__
#else
#define GLCM_JAX_INLINE_OR_DEVICE inline
#endif

template <typename T>
GLCM_JAX_INLINE_OR_DEVICE void addition(const T& x0, const T& x1, T* out) {

  *out = x0 + x1;

}

}  // namespace glcm_jax

#endif
