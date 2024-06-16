// Replace the kernel with an already implemented saxpy from cublas library.

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void saxpy(int n, float a, float *x, float *y)
{
    float *d_x, *d_y;
    cublasHandle_t handle;

    // Initialize CUBLAS context
    cublasCreate(&handle);

    // Allocate GPU memory
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Call cublasSaxpy
    // cublasSaxpy(handle, n, &a, d_x, 1, d_y, 1);
    cublasSaxpy(handle, n, &a, d_x, 1, d_y, 1);

    // Copy result back to host
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);
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

    // Call SAXPY function using cuBLAS
    saxpy(n, a, x, y);

    // Free host memory
    free(x);
    free(y);

    return 0;
}
