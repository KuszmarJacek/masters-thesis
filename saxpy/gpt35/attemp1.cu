// Given below code, rewrite it in CUDA.
// void saxpy(int n, float a, float *x, float *y)
// {
// for (int i = 0; i < n; ++i)
// y[i] = a * x[i] + y[i];
// }

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

void saxpy(int n, float a, float *x, float *y) {
    const int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    saxpy_kernel<<<numBlocks, blockSize>>>(n, a, x, y);
    cudaDeviceSynchronize(); // Ensure all threads finish before returning
}
