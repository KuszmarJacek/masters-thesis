// Optimize provided kernel by reducing the final warp with shfl\_down
// operation from the cooperative groups:
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

template <typename T>
__global__ void reduce2(T *g_idata, T *g_odata, unsigned int n) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s = cta.size() / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }

    // Warp-level reduction using shuffle operation
    T val = sdata[0];
    for (int offset = cta.size() / 2; offset > 0; offset /= 2) {
        val += cg::shfl_down(val, offset);
    }

    // Store the warp reduced value back to shared memory
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = val;
    }

    cg::sync(cta);

    // Write the warp reduced value to global memory
    if (tid < warpSize) {
        g_odata[blockIdx.x] = sdata[tid];
    }
}