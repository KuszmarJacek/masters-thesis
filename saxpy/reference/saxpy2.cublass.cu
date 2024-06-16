#include <stdio.h>
#include <cublas_v2.h>

int main(void)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    int n = 1 << 20;
    float *x, *y, alpha = 2.0;
    cudaMalloc(&x, n * sizeof(float));
    cudaMalloc(&y, n * sizeof(float));
    cublasSaxpy(handle, n, &alpha, x, 1, y, 1);
    cublasDestroy(handle);
    cudaFree(x);
    cudaFree(y);
}