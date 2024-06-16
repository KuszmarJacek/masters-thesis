// Rewrite provided sequential code in C to a CUDA kernel. Assume that n = 1 << 20;
// void saxpy(int n, float a, float *x, float *y)
// {
//     for (int i = 0; i < n; ++i)
//         y[i] = a * x[i] + y[i];
// }

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void saxpy_kernel(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

void saxpy(int n, float a, float *x, float *y)
{
    float *d_x, *d_y;

    // Allocate GPU memory
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Execution configuration
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (n + blockSize - 1) / blockSize;  // Number of blocks

    // Launch the kernel
    saxpy_kernel<<<numBlocks, blockSize>>>(n, a, d_x, d_y);

    // Copy result back to host
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
}

int main()
{
    int n = 1 << 20;  // Example size
    float a = 2.0f;
    float *x, *y;

    // Allocate memory on the host
    x = (float*)malloc(n * sizeof(float));
    y = (float*)malloc(n * sizeof(float));

    // Initialize data
    for(int i = 0; i < n; i++) {
        x[i] = 1.0f;  // Example values
        y[i] = 2.0f;
    }

    // Call SAXPY function
    saxpy(n, a, x, y);

    // Free host memory
    free(x);
    free(y);

    return 0;
}
