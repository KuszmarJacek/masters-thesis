#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include <chrono>


__global__ void scanExclusiveKernel(uint *dst, uint *src, uint arrayLength) {
  // Get the index of the current array based on block index
  uint i = blockIdx.x;

  // Compute the starting index for src and dst arrays for this block
  uint startIdx = i * arrayLength;
  src += startIdx;
  dst += startIdx;

  // Thread index within the block represents the element index within the array
  uint j = threadIdx.x;

  // Shared memory for inter-thread communication within the block
  extern __shared__ uint temp[];

  // Load input into shared memory for faster access
  if (j < arrayLength) {
    temp[j] = src[j];
  }
  __syncthreads(); // Ensure all data is loaded into shared memory

  // Initialize the first element of the block's output to 0
  if (j == 0) {
    dst[0] = 0;
  }

  // Perform scan using shared memory
  if (j < arrayLength) {
    uint sum = 0;
    for (int k = 1; k <= j; k++) {
      sum += temp[k - 1];
    }
    dst[j] = sum;
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


#define THREADBLOCK_SIZE 256

int main() {
    // Number of arrays and length of each array
    uint batchSize = 256;
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

    // Launch the kernel
    scanExclusiveKernel<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE),
                        THREADBLOCK_SIZE>>>(d_dst, d_src, arrayLength);
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
