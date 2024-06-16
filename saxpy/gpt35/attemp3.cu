// Rewrite your kernel to use a grid stride loop
__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = i; idx < n; idx += stride) {
        y[idx] = a * x[idx] + y[idx];
    }
}
