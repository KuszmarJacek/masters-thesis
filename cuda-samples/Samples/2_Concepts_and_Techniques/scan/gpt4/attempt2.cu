#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Define a basic parallel exclusive scan kernel for segments of the array
__global__ void scanExclusiveLargeKernel(uint *dst, uint *src, uint arrayLength, uint maxThreadsPerBlock) {
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint i = blockId * (maxThreadsPerBlock * 2) + threadIdx.x;
    uint tid = threadIdx.x;

    extern __shared__ uint temp[]; // Allocated dynamically at kernel launch

    // Load data into shared memory in two phases to handle larger than block size data
    if (i < arrayLength) {
        temp[tid] = src[i];
    } else {
        temp[tid] = 0;
    }
    if (i + maxThreadsPerBlock < arrayLength) {
        temp[tid + maxThreadsPerBlock] = src[i + maxThreadsPerBlock];
    } else {
        temp[tid + maxThreadsPerBlock] = 0;
    }
    __syncthreads();

    // Perform scan on the data in shared memory
    for (uint stride = 1; stride <= maxThreadsPerBlock; stride *= 2) {
        uint index = (tid + 1) * stride * 2 - 1;
        if (index < 2 * maxThreadsPerBlock) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // Reverse phase of the scan
    for (uint stride = maxThreadsPerBlock / 2; stride > 0; stride /= 2) {
        __syncthreads();
        uint index = (tid + 1) * stride * 2 - 1;
        if (index + stride < 2 * maxThreadsPerBlock) {
            temp[index + stride] += temp[index];
        }
    }
    __syncthreads();

    // Write results back to global memory
    if (i < arrayLength) {
        dst[i] = (tid > 0) ? temp[tid - 1] : 0;
    }
    if (i + maxThreadsPerBlock < arrayLength) {
        dst[i + maxThreadsPerBlock] = temp[tid + maxThreadsPerBlock - 1];
    }
}

void scanExclusiveHost(uint *dst, uint *src, uint batchSize, uint arrayLength) {
    for (uint i = 0; i < batchSize; i++, src += arrayLength, dst += arrayLength) {
        dst[0] = 0;
        for (uint j = 1; j < arrayLength; j++) {
            dst[j] = src[j - 1] + dst[j - 1];
        }
    }
}

int main() {
    // Number of arrays and length of each array
    uint batchSize = 100;  // Increased batch size for testing
    uint arrayLength = 10240;  // Increased array length for testing

    // Allocate memory for host arrays
    uint *h_src = new uint[batchSize * arrayLength];
    uint *h_dst_gpu = new uint[batchSize * arrayLength];
    uint *h_dst_host = new uint[batchSize * arrayLength];

    // Initialize source data
    for (uint i = 0; i < batchSize * arrayLength; ++i) {
        h_src[i] = rand() % 10;  // Random numbers for example
    }

    // Allocate memory for device arrays
    uint *d_src, *d_dst;
    cudaMalloc(&d_src, batchSize * arrayLength * sizeof(uint));
    cudaMalloc(&d_dst, batchSize * arrayLength * sizeof(uint));

    // Copy source data from host to device
    cudaMemcpy(d_src, h_src, batchSize * arrayLength * sizeof(uint), cudaMemcpyHostToDevice);

    uint maxThreadsPerBlock = 256;
    uint numElementsPerBlock = maxThreadsPerBlock * 2;
    uint numBlocks = (arrayLength + numElementsPerBlock - 1) / numElementsPerBlock;

    dim3 grid(numBlocks, batchSize); // 2D grid to handle large arrayLength per batch

    // Measure kernel execution with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the kernel
    scanExclusiveLargeKernel<<<grid, maxThreadsPerBlock, 2 * maxThreadsPerBlock * sizeof(uint)>>>(d_dst, d_src, arrayLength, maxThreadsPerBlock);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Kernel Time: " << milliseconds << " ms" << std::endl;

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    cudaMemcpy(h_dst_gpu, d_dst, batchSize * arrayLength * sizeof(uint), cudaMemcpyDeviceToHost);

    // Measure host function time using std::chrono
    auto host_start = std::chrono::high_resolution_clock::now();
    scanExclusiveHost(h_dst_host, h_src, batchSize, arrayLength);
    auto host_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> host_elapsed = host_stop - host_start;
    std::cout << "Host Function Time: " << host_elapsed.count() << " ms" << std::endl;

    // Compare results
    bool correct = true;
    for (uint i = 0; i < batchSize * arrayLength; ++i) {
        if (h_dst_gpu[i] != h_dst_host[i]) {
            std::cerr << "Mismatch at index " << i << ": GPU " << h_dst_gpu[i] << ", Host " << h_dst_host[i] << std::endl;
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "Test Passed: GPU results match the host results." << std::endl;
    } else {
        std::cout << "Test Failed: GPU results do not match the host results." << std::endl;
    }

    // Free device memory
    cudaFree(d_src);
    cudaFree(d_dst);

    // Free host memory
    delete[] h_src;
    delete[] h_dst_gpu;
    delete[] h_dst_host;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
