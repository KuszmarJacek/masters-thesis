// Rewrite this function as a CUDA kernel
// template <class T>
// T reduceCPU(T *data, int size) {
//   T sum = data[0];
//   T c = (T)0.0;

//   for (int i = 1; i < size; i++) {
//     T y = data[i] - c;
//     T t = sum + y;
//     c = (t - sum) - y;
//     sum = t;
//   }

//   return sum;
// }

#include <cuda_runtime.h>

template <class T>
__global__ void reduceKernel(T *data, T *result, int size) {
    extern __shared__ T sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < size) {
        sdata[tid] = data[i];
    } else {
        sdata[tid] = (T)0;  // Handle case where size is not a multiple of blockDim.x
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

// To be called from the host
template <class T>
T reduce(T *data, int size) {
    int blockSize = 1024;  // Number of threads in each block
    int numBlocks = (size + blockSize - 1) / blockSize;
    T *d_data, *d_result;
    T *partialSums = new T[numBlocks];

    cudaMalloc(&d_data, size * sizeof(T));
    cudaMalloc(&d_result, numBlocks * sizeof(T));
    cudaMemcpy(d_data, data, size * sizeof(T), cudaMemcpyHostToDevice);

    reduceKernel<<<numBlocks, blockSize, blockSize * sizeof(T)>>>(d_data, d_result, size);

    // Copy the results back
    cudaMemcpy(partialSums, d_result, numBlocks * sizeof(T), cudaMemcpyDeviceToHost);

    // Finish reduction on CPU
    T sum = 0;
    for (int i = 0; i < numBlocks; i++) {
        sum += partialSums[i];
    }

    cudaFree(d_data);
    cudaFree(d_result);
    delete[] partialSums;

    return sum;
}
