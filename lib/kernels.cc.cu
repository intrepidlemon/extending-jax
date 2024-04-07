// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point accross.

#include "addition.h"
#include "kernel_helpers.h"
#include "kernels.h"

namespace glcm_jax {

namespace {

template <typename T>
__global__ void glcm_kernel(std::int64_t size, const T *x0, const T *x1, T *out) {
  for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += blockDim.x * gridDim.x) {
    addition<T>(x0[idx], x1[idx], out + idx);
  }
}

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <typename T>
inline void apply_glcm(cudaStream_t stream, void **buffers, const char *opaque,
                         std::size_t opaque_len) {
  const GLCMDescriptor &d = *UnpackDescriptor<GLCMDescriptor>(opaque, opaque_len);
  const std::int64_t size = d.size;

  const T *x0 = reinterpret_cast<const T *>(buffers[0]);
  const T *x1 = reinterpret_cast<const T *>(buffers[1]);
  T *out = reinterpret_cast<T *>(buffers[2]);

  const int block_dim = 128;
  const int grid_dim = std::min<int>(1024, (size + block_dim - 1) / block_dim);
  glcm_kernel<T>
      <<<grid_dim, block_dim, 0, stream>>>(size, x0, x1, out);

  ThrowIfError(cudaGetLastError());
}

}  // namespace

void gpu_glcm_f32(cudaStream_t stream, void **buffers, const char *opaque,
                    std::size_t opaque_len) {
  apply_glcm<float>(stream, buffers, opaque, opaque_len);
}

void gpu_glcm_f64(cudaStream_t stream, void **buffers, const char *opaque,
                    std::size_t opaque_len) {
  apply_glcm<double>(stream, buffers, opaque, opaque_len);
}

}  // namespace glcm_jax
