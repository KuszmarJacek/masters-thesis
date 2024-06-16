/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample calculates scalar products of a
 * given set of input vector pairs
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <chrono>

///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU
///////////////////////////////////////////////////////////////////////////////
extern "C" void scalarProdCPU(float *h_C, float *h_A, float *h_B, int vectorN,
                              int elementN);

///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on GPU
///////////////////////////////////////////////////////////////////////////////
#include "scalarProd_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////

// Total number of input vector pairs; arbitrary
const int VECTOR_N = 4096;
// Number of elements per vector; arbitrary,
// but strongly preferred to be a multiple of warp size
// to meet memory coalescing constraints
const int ELEMENT_N = 16384;
// Total number of data elements
const int DATA_N = VECTOR_N * ELEMENT_N;

const int DATA_SZ = DATA_N * sizeof(float);
const int RESULT_SZ = VECTOR_N * sizeof(float);

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    float *h_A, *h_B, *h_C_CPU, *h_C_GPU, *h_C_GPU_GPT;
    float *d_A, *d_B, *d_C, *d_C_GPT;
    double delta, ref, sum_delta, sum_ref, L1norm, L1norm_GPT;
    StopWatchInterface *hTimer = NULL;
    int i;

    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest
    // Gflops/s
    findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&hTimer);

    printf("Initializing data...\n");
    printf("...allocating CPU memory.\n");
    h_A = (float *)malloc(DATA_SZ);
    h_B = (float *)malloc(DATA_SZ);
    h_C_CPU = (float *)malloc(RESULT_SZ);
    h_C_GPU = (float *)malloc(RESULT_SZ);
    h_C_GPU_GPT = (float *)malloc(RESULT_SZ);

    printf("...allocating GPU memory.\n");
    checkCudaErrors(cudaMalloc((void **)&d_A, DATA_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_B, DATA_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_C, RESULT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_C_GPT, RESULT_SZ));

    printf("...generating input data in CPU mem.\n");
    srand(123);

    // Generating input data on CPU
    for (i = 0; i < DATA_N; i++)
    {
        h_A[i] = RandFloat(0.0f, 1.0f);
        h_B[i] = RandFloat(0.0f, 1.0f);
    }

    // Original GPU kernel
    printf("...copying input data to GPU mem.\n");
    // Copy options data to GPU memory for further processing
    checkCudaErrors(cudaMemcpy(d_A, h_A, DATA_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, DATA_SZ, cudaMemcpyHostToDevice));
    printf("Data init done.\n");

    printf("Executing GPU kernel...\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    scalarProdGPU<<<128, 256>>>(d_C, d_A, d_B, VECTOR_N, ELEMENT_N);
    getLastCudaError("scalarProdGPU() execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    printf("GPU time: %f msecs.\n", sdkGetTimerValue(&hTimer));

    printf("Reading back GPU result...\n");
    // Read back GPU results to compare them to CPU results
    checkCudaErrors(cudaMemcpy(h_C_GPU, d_C, RESULT_SZ, cudaMemcpyDeviceToHost));

    printf("Executing GPU GPT kernel...\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);



    // GPT4scalarProdGPU<<<128, 256>>>(d_C_GPT, d_A, d_B, VECTOR_N, ELEMENT_N);
    // GPT4scalarProdGPUOptimized<<<128, 256>>>(d_C_GPT, d_A, d_B, VECTOR_N, ELEMENT_N);
    // GPT35scalarProdCUDA<<<128, 256>>>(d_C_GPT, d_A, d_B, VECTOR_N, ELEMENT_N);
    // GPT35scalarProdCUDAOptimized<<<128, 256>>>(d_C_GPT, d_A, d_B, VECTOR_N, ELEMENT_N);

    GeminiScalarProdGPU<<<128, 256>>>(d_C_GPT, d_A, d_B, VECTOR_N, ELEMENT_N);
    
    

    getLastCudaError("scalarProdGPU() execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    printf("GPU GPT time: %f msecs.\n", sdkGetTimerValue(&hTimer));

    printf("Reading back GPU GPT result...\n");
    // Read back GPU results to compare them to CPU results
    checkCudaErrors(cudaMemcpy(h_C_GPU_GPT, d_C_GPT, RESULT_SZ, cudaMemcpyDeviceToHost));

    printf("Checking GPU results...\n");
    printf("..running CPU scalar product calculation\n");
    auto host_start = std::chrono::high_resolution_clock::now();
    scalarProdCPU(h_C_CPU, h_A, h_B, VECTOR_N, ELEMENT_N);
    auto host_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> host_elapsed = host_stop - host_start;
    std::cout << "Host Function Time: " << host_elapsed.count() << " ms" << std::endl;

    printf("...comparing the results for non GPT\n");

    sum_delta = 0;
    sum_ref = 0;

    for (i = 0; i < VECTOR_N; i++)
    {
        delta = fabs(h_C_GPU[i] - h_C_CPU[i]);
        ref = h_C_CPU[i];
        sum_delta += delta;
        sum_ref += ref;
    }

    L1norm = sum_delta / sum_ref;

    printf("...comparing the results for GPT\n");
    sum_delta = 0;
    sum_ref = 0;

    for (i = 0; i < VECTOR_N; i++)
    {
        delta = fabs(h_C_GPU_GPT[i] - h_C_CPU[i]);
        ref = h_C_CPU[i];
        sum_delta += delta;
        sum_ref += ref;
    }

    L1norm_GPT = sum_delta / sum_ref;

    printf("Shutting down...\n");
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_A));
    free(h_C_GPU);
    free(h_C_GPU_GPT);
    free(h_C_CPU);
    free(h_B);
    free(h_A);
    sdkDeleteTimer(&hTimer);

    printf("L1 error: %E\n", L1norm);
    printf((L1norm < 1e-6) ? "Test passed\n" : "Test failed!\n");
    printf("L1 GPT error: %E\n", L1norm_GPT);
    printf((L1norm_GPT < 1e-6) ? "Test passed\n" : "Test failed!\n");
}
