#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include <chrono>

#define THREADBLOCK_SIZE 256

__global__ void scanExclusiveKernel1(uint *dst, uint *src, uint arrayLength) {
    uint batchIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;

    for (uint i = batchIndex; i < arrayLength; i += stride) {
        dst[i] = 0;
        for (uint j = 1; j < arrayLength; j++) {
            dst[i * arrayLength + j] = src[i * arrayLength + j - 1] + dst[i * arrayLength + j - 1];
        }
    }
}

__global__ void scanExclusiveKernel2(uint *dst, uint *src, uint batchSize, uint arrayLength) {
    uint batchIndex = blockIdx.y * gridDim.x + blockIdx.x;
    uint stride = gridDim.x * gridDim.y * blockDim.x;

    for (uint i = batchIndex; i < batchSize; i += stride) {
        uint offset = i * arrayLength;

        dst[offset] = 0;
        for (uint j = 1; j < arrayLength; j++) {
            dst[offset + j] = src[offset + j - 1] + dst[offset + j - 1];
        }
    }
}


// Sequential version of the exclusive scan for verification
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
    uint batchSize = 256;
    // uint threadsPerBlock = 256;
    // uint arrayLength = 607;
    // uint arrayLength = 16384;
    // uint arrayLength = 16384 * 2;
    // uint arrayLength = 4096;
    uint arrayLength = 1024;

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

    // Declare CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording the GPU time
    cudaEventRecord(start);

    // scanExclusiveKernel1<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE),
    //                     THREADBLOCK_SIZE>>>(d_dst, d_src, 4 * THREADBLOCK_SIZE);

    // scanExclusiveKernel1<<<(batchSize + threadsPerBlock - 1) / threadsPerBlock, arrayLength>>>(d_dst, d_src, arrayLength);

    // scanExclusiveKernel1<<<batchSize, arrayLength, 1024 * sizeof(uint)>>>(d_dst, d_src, arrayLength);

    // ----------------------------------------------------------------------------------------------------------------------


    scanExclusiveKernel2<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE),
                        THREADBLOCK_SIZE>>>(d_dst, d_src, batchSize, arrayLength);

    cudaDeviceSynchronize();

    // Stop recording the GPU time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Kernel Time: " << milliseconds << " ms" << std::endl;

    // Check for kernel errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    // Copy results back to host
    cudaMemcpy(h_dst_gpu, d_dst, batchSize * arrayLength * sizeof(uint), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // Measure host execution time
    auto host_start = std::chrono::high_resolution_clock::now();
    
    // Perform the sequential scan on the host
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

