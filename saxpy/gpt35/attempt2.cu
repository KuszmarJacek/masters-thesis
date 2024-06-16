// provide a working example with memory initialization

#include <iostream>

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

void saxpy(int n, float a, float *x, float *y) {
    const int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    saxpy_kernel<<<numBlocks, blockSize>>>(n, a, x, y);
}

int main() {
    int n = 10000;
    float a = 2.0f;
    
    // Allocate memory on the host
    float *h_x = new float[n];
    float *h_y = new float[n];

    // Initialize arrays
    for (int i = 0; i < n; ++i) {
        h_x[i] = i * 0.5f;
        h_y[i] = i * 1.5f;
    }

    // Allocate memory on the device
    float *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    saxpy(n, a, d_x, d_y);

    // Copy result back to host
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Output result
    std::cout << "Result after SAXPY operation:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "y[" << i << "] = " << h_y[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);

    // Free host memory
    delete[] h_x;
    delete[] h_y;

    return 0;
}
