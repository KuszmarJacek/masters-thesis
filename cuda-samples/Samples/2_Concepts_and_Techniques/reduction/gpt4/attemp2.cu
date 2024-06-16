// Optimize provided kernel by reducing the final warp with shfl_down operation from the cooperative groups
// template <class T>
// __global__ void reduce2(T *g_idata, T *g_odata, unsigned int n) {
//   // Handle to thread block group
//   cg::thread_block cta = cg::this_thread_block();
//   T *sdata = SharedMemory<T>();

//   // load shared mem
//   unsigned int tid = threadIdx.x;
//   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

//   sdata[tid] = (i < n) ? g_idata[i] : 0;

//   cg::sync(cta);

//   // do reduction in shared mem
//   for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//     if (tid < s) {
//       sdata[tid] += sdata[tid + s];
//     }

//     cg::sync(cta);
//   }

//   // write result for this block to global mem
//   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
// }

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <class T>
__device__ T* SharedMemory() {
    // Externally provided shared memory
    extern __shared__ int s[];
    return reinterpret_cast<T*>(s);
}

template <class T>
__global__ void reduce2(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory, initializing to zero if out of range
  sdata[tid] = (i < n) ? g_idata[i] : 0;
  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    cg::sync(cta);
  }

  // Use warp shuffle operations for the last warp reduction
  if (tid < 32) {
    T val = sdata[tid];
    // Perform warp-wide reduction using shuffle operations
    for (int offset = 16; offset > 0; offset /= 2) {
      val += cg::shfl_down(cta, val, offset);
    }
    sdata[tid] = val;  // Write the reduced value back to shared memory
  }

  // Write the result for this block to global memory
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}