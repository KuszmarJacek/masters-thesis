// Replace the kernel with a call to saxpy from the cuBLAS library

#include <iostream>
#include <cublas_v2.h>

void saxpy(int n, float a, float *x, float *y) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSaxpy(handle, n, &a, x, 1, y, 1);
    cublasDestroy(handle);
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

    // Call saxpy from cuBLAS
    saxpy(n, a, h_x, h_y);

    // Output result
    std::cout << "Result after SAXPY operation:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "y[" << i << "] = " << h_y[i] << std::endl;
    }

    // Free host memory
    delete[] h_x;
    delete[] h_y;

    return 0;
}
