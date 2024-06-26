#ifndef _KEPLER_JAX_KERNELS_H_
#define _KEPLER_JAX_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace glcm_jax {
struct GLCMDescriptor {
  std::int32_t size;
};

void gpu_glcm_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);
void gpu_glcm_f64(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

}  // namespace glcm_jax

#endif
