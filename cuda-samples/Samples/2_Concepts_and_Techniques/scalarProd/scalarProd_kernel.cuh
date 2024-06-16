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

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

///////////////////////////////////////////////////////////////////////////////
// On G80-class hardware 24-bit multiplication takes 4 clocks per warp
// (the same as for floating point  multiplication and addition),
// whereas full 32-bit multiplication takes 16 clocks per warp.
// So if integer multiplication operands are  guaranteed to fit into 24 bits
// (always lie within [-8M, 8M - 1] range in signed case),
// explicit 24-bit multiplication is preferred for performance.
///////////////////////////////////////////////////////////////////////////////
#define IMUL(a, b) __mul24(a, b)

///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on GPU
// Parameters restrictions:
// 1) ElementN is strongly preferred to be a multiple of warp size to
//    meet alignment constraints of memory coalescing.
// 2) ACCUM_N must be a power of two.
///////////////////////////////////////////////////////////////////////////////
#define ACCUM_N 1024

__global__ void scalarProdGPU(float *d_C, float *d_A, float *d_B, int vectorN,
                              int elementN) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Accumulators cache
  __shared__ float accumResult[ACCUM_N];

  ////////////////////////////////////////////////////////////////////////////
  // Cycle through every pair of vectors,
  // taking into account that vector counts can be different
  // from total number of thread blocks
  ////////////////////////////////////////////////////////////////////////////
  for (int vec = blockIdx.x; vec < vectorN; vec += gridDim.x) {
    int vectorBase = IMUL(elementN, vec);
    int vectorEnd = vectorBase + elementN;

    ////////////////////////////////////////////////////////////////////////
    // Each accumulator cycles through vectors with
    // stride equal to number of total number of accumulators ACCUM_N
    // At this stage ACCUM_N is only preferred be a multiple of warp size
    // to meet memory coalescing alignment constraints.
    ////////////////////////////////////////////////////////////////////////
    for (int iAccum = threadIdx.x; iAccum < ACCUM_N; iAccum += blockDim.x) {
      float sum = 0;

      for (int pos = vectorBase + iAccum; pos < vectorEnd; pos += ACCUM_N)
        sum += d_A[pos] * d_B[pos];

      accumResult[iAccum] = sum;
    }

    ////////////////////////////////////////////////////////////////////////
    // Perform tree-like reduction of accumulators' results.
    // ACCUM_N has to be power of two at this stage
    ////////////////////////////////////////////////////////////////////////
    for (int stride = ACCUM_N / 2; stride > 0; stride >>= 1) {
      cg::sync(cta);

      for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x)
        accumResult[iAccum] += accumResult[stride + iAccum];
    }

    cg::sync(cta);

    if (threadIdx.x == 0) d_C[vec] = accumResult[0];
  }
}

__global__ void GPT4scalarProdGPU(float *d_C, float *d_A, float *d_B, int VECTOR_N, int ELEMENT_N) {
    int vec = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec < VECTOR_N) {
        double sum = 0;
        int vectorBase = ELEMENT_N * vec;
        int vectorEnd = vectorBase + ELEMENT_N;

        for (int pos = vectorBase; pos < vectorEnd; pos++) {
            sum += d_A[pos] * d_B[pos];
        }

        d_C[vec] = (float)sum;
    }
}

__global__ void GPT4scalarProdGPUOptimized(float *d_C, float *d_A, float *d_B, int VECTOR_N, int ELEMENT_N) {
    int vec = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec < VECTOR_N) {
        float sum = 0;  // Use float for faster computation if double precision is not needed
        int vectorBase = ELEMENT_N * vec;
        int vectorEnd = vectorBase + ELEMENT_N;

        // Use a smaller loop with unrolling for better performance
        for (int pos = vectorBase; pos < vectorEnd; pos += 4) {
            sum += d_A[pos] * d_B[pos];
            if (pos + 1 < vectorEnd) sum += d_A[pos + 1] * d_B[pos + 1];
            if (pos + 2 < vectorEnd) sum += d_A[pos + 2] * d_B[pos + 2];
            if (pos + 3 < vectorEnd) sum += d_A[pos + 3] * d_B[pos + 3];
        }

        d_C[vec] = sum;
    }
}

__global__ void GPT35scalarProdCUDA(float *d_C, float *d_A, float *d_B, int vectorN, int elementN) {
    int vec = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec < vectorN) {
        int vectorBase = elementN * vec;
        int vectorEnd = vectorBase + elementN;

        double sum = 0;

        for (int pos = vectorBase; pos < vectorEnd; pos++)
            sum += d_A[pos] * d_B[pos];

        d_C[vec] = (float)sum;
    }
}

#define BLOCK_SIZE 1024

__global__ void GPT35scalarProdCUDAOptimized(float *d_C, float *d_A, float *d_B, int vectorN, int elementN) {
    __shared__ float shared_A[BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE];

    int vec = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    int offset = vec * elementN;

    // Load data into shared memory
    for (int i = tid; i < elementN; i += blockDim.x) {
        shared_A[i] = d_A[offset + i];
        shared_B[i] = d_B[offset + i];
    }
    __syncthreads();

    // Compute dot product
    for (int i = 0; i < elementN; i++) {
        sum += shared_A[i] * shared_B[i];
    }
    
    // Store the result
    if (vec < vectorN) {
        d_C[vec] = sum;
    }
}

__global__ void GeminiScalarProdGPU(float *d_C, float *d_A, float *d_B, int vectorN, int elementN) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < vectorN * elementN) {
    int vec = i / elementN;
    int pos = i % elementN;
    float sum = 0.0f;

    // Unroll loop for potential performance gains (adjust unroll factor as needed)
    for (int unroll = 0; unroll < 4; unroll++) {
      if (i + unroll * elementN < vectorN * elementN) {
        sum += d_A[i + unroll * elementN] * d_B[i + unroll * elementN];
      }
    }

    d_C[vec] = sum;
  }
}