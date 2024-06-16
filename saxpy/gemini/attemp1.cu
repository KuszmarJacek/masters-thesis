// given below code, rewrite it in cuda
// void saxpy(int n, float a, float *x, float *y) { for (int i = 0; i < n; ++i) y[i] = a * x[i] + y[i]; }


#include <cuda.h>
#include <cuda_runtime.h>

__global__ void saxpy_cuda(int n, float a, float *d_x, float *d_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        atomicAdd(&d_y[i], a * d_x[i]);
    }
}

int main(int argc, char* argv[]) {
    int N = 1 << 20;  // Adjust as needed (power of 2 for efficiency)
    float *x, *y;
    float *d_x, *d_y;

    // Allocate host memory
    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    // Initialize host arrays (optional, can be done on device)
    for (int i = 0; i < N; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Determine optimal block size based on device architecture
    int block_size = getOptimalBlockSize(N);  // Replace with your function

    // Launch the kernel
    saxpy_cuda<<<(N + block_size - 1) / block_size, block_size>>>(N, 2.0f, d_x, d_y);

    // Error checking (optional)
    cudaDeviceSynchronize();  // Wait for kernel to finish

    // Copy results back from device to host
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);

    // Free host memory
    free(x);
    free(y);

    // Process the results in y (if needed)

    return 0;
}
