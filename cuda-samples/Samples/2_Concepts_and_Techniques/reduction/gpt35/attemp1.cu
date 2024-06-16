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

template <typename T>
__global__ void reduceGPU(T *data, T *result, int size) {
    __shared__ T sdata[256]; // Assuming 256 threads per block

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? data[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

template <typename T>
T reduceGPUWrapper(T *data, int size) {
    const int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    T *d_result;
    cudaMalloc((void **)&d_result, numBlocks * sizeof(T));

    reduceGPU<<<numBlocks, blockSize>>>(data, d_result, size);

    T *result = (T *)malloc(numBlocks * sizeof(T));
    cudaMemcpy(result, d_result, numBlocks * sizeof(T), cudaMemcpyDeviceToHost);

    T finalResult = result[0];
    for (int i = 1; i < numBlocks; i++) {
        finalResult += result[i];
    }

    free(result);
    cudaFree(d_result);

    return finalResult;
}
