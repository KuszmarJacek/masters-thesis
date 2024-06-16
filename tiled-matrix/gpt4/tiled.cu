#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "time.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#define BLOCK_SIZE 16
#define TILE_WIDTH 16

/*
 * prints matrices
 * Because matrices filled with dummy 0s function takes 3 dim arguments:
 *      actual x and y dimension and dim as big square matrix's dimension
 */
void print_matrices(float* matrix, char* file_Name, int x_dim, int y_dim, int dim)
{
    std::ofstream outFile;
    outFile.open (file_Name);

    outFile << std::fixed;
    outFile << std::setprecision(2);

    for (int i = 0; i < x_dim; i++) {

        for (int j = 0; j < y_dim; j++) {
            outFile << matrix[i * dim + j] << " ";
        }
        outFile << std::endl;
    }
}

//naive CPU matrix multiplication code
//because of its simplicity directly taken from web
//it multiplies square matrices
__host__ void cpu_matrix_mult(float *h_a, float *h_b, float *h_result, int m) {
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            float tmp = 0.0;
            for (int h = 0; h < m; ++h)
            {
                tmp += h_a[i * m + h] * h_b[h * m + j];
            }
            h_result[i * m + j] = tmp;
        }
    }
}

//this function is for filling the matrices with cos and sin values randomly
//I transform the matrices to square matrix in order to perform better multiplication
__host__ int fill(float **Lmatrix, float **Rmatrix, int LdimX, int LdimY, int RdimX, int RdimY) {

    int sqr_dim_X, sqr_dim_Y, size;

    sqr_dim_X = RdimX;
    if (LdimX > RdimX) {
        sqr_dim_X = LdimX;
    }

    sqr_dim_Y = RdimY;
    if (LdimY > RdimY) {
        sqr_dim_Y = LdimY;
    }

    size = sqr_dim_Y;
    if (sqr_dim_X > sqr_dim_Y) {
        size = sqr_dim_X;
    }

    int temp = size / BLOCK_SIZE + (size % BLOCK_SIZE == 0 ? 0 : 1);
    size = temp * BLOCK_SIZE;

    size_t pt_size = size * size * sizeof(float);

    *Lmatrix = (float *) malloc(pt_size);
    *Rmatrix = (float *) malloc(pt_size);

    memset(*Lmatrix, 0, pt_size);
    memset(*Rmatrix, 0, pt_size);

    for (int i = 0; i < LdimX; i++) {
        for (int j = 0; j < LdimY; j++) {
            int dummy = size * i + j;
            (*Lmatrix)[dummy] = sinf(dummy);
        }
    }
    for (int i = 0; i < RdimX; i++) {
        for (int j = 0; j < RdimY; j++) {
            int dummy = size * i + j;
            (*Rmatrix)[dummy] = cosf(dummy);
        }
    }
    return size;
}

__global__ void GPT4matrixMultTiled(float *d_a, float *d_b, float *d_result, int m) {
    // Allocate 2D tiles in shared memory
    __shared__ float tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_b[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float tmp = 0.0;

    // Loop over the tiles of the input matrices
    for (int k = 0; k < (m - 1) / TILE_WIDTH + 1; ++k) {
        // Load the tiles into shared memory
        if (row < m && (k*TILE_WIDTH + threadIdx.x) < m)
            tile_a[threadIdx.y][threadIdx.x] = d_a[row * m + k * TILE_WIDTH + threadIdx.x];
        else
            tile_a[threadIdx.y][threadIdx.x] = 0.0;

        if (col < m && (k*TILE_WIDTH + threadIdx.y) < m)
            tile_b[threadIdx.y][threadIdx.x] = d_b[(k * TILE_WIDTH + threadIdx.y) * m + col];
        else
            tile_b[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        // Perform the multiplication for the tile
        for (int n = 0; n < TILE_WIDTH; ++n) {
            tmp += tile_a[threadIdx.y][n] * tile_b[n][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the result to global memory
    if (row < m && col < m)
        d_result[row * m + col] = tmp;
}

// main routine that executes on the host
int main(void)
{
    //size of the vectors to be processed  and matrix dimensions
    int Left_matrix_x  = 512; 
    int Left_matrix_y  = 512; 
    int Right_matrix_x = 512; 
    int Right_matrix_y = 512;

    // int Left_matrix_x  = 1024; 
    // int Left_matrix_y  = 1024; 
    // int Right_matrix_x = 1024; 
    // int Right_matrix_y = 1024; 

    // int Left_matrix_x  = 1024 * 2; 
    // int Left_matrix_y  = 1024 * 2; 
    // int Right_matrix_x = 1024 * 2; 
    // int Right_matrix_y = 1024 * 2; 

    // int Left_matrix_x  = 1024 * 4; 
    // int Left_matrix_y  = 1024 * 4; 
    // int Right_matrix_x = 1024 * 4; 
    // int Right_matrix_y = 1024 * 4; 


    int Left_vector_size;
    int Right_vector_size;

    float *Left_Vector_h, *Right_Vector_h, *Left_Vector_d, *Right_Vector_d, *Res_h, *Res_d, *CPU;  // Pointer to host & device arrays

    // printf("Enter m n n k :\n");

    // scanf("%d %d %d %d",&Left_matrix_x,&Left_matrix_y,&Right_matrix_x,&Right_matrix_y); // input matrix dimensions are taken

    int dim = fill(&Left_Vector_h, &Right_Vector_h, Left_matrix_x, Left_matrix_y, Right_matrix_x, Right_matrix_y); //fills the matrices with random values

    print_matrices(Left_Vector_h,"Input_LHS",Left_matrix_x,Left_matrix_y,dim);
    print_matrices(Right_Vector_h,"Input_RHS",Right_matrix_x,Right_matrix_y,dim);

    size_t vector_size;
    vector_size = dim*dim * sizeof(float);

    Res_h = (float *) malloc(vector_size); // Allocate array on host for result
    CPU = (float *) malloc(vector_size);// Allocate array on host for CPU_matrix_multiplication result

    cudaMalloc((void **) &Left_Vector_d, vector_size);     // Allocate array on device for LHS operand
    cudaMalloc((void **) &Right_Vector_d, vector_size);   // Allocate array on device for RHS operand but this is vector 1xN
    cudaMalloc((void **) &Res_d, vector_size);     // Allocate array on device for result

    cudaMemcpy(Left_Vector_d, Left_Vector_h, vector_size, cudaMemcpyHostToDevice);      // copy values to device
    cudaMemcpy(Right_Vector_d, Right_Vector_h, vector_size, cudaMemcpyHostToDevice);   // copy values to device

    //Block dimension is directly from block_size
    dim3 Block_dim(BLOCK_SIZE, BLOCK_SIZE);
    //Grid dimension is found by dividing matrix dimension to block_size
    dim3 Grid_dim(dim / BLOCK_SIZE, dim / BLOCK_SIZE);

    //commented out the functions which helps to calculate time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    //kernel call
    GPT4matrixMultTiled<< < Grid_dim, Block_dim >> > (Left_Vector_d, Right_Vector_d, Res_d, dim);

    //commented out the functions which helps to calculate time
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float et;
    cudaEventElapsedTime(&et, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Retrieve result from device and store it in host array
    cudaMemcpy(Res_h, Res_d, vector_size, cudaMemcpyDeviceToHost);

    clock_t begin = clock();

    cpu_matrix_mult(Left_Vector_h,Right_Vector_h,CPU,dim); //matrix multiplication on cpu

    clock_t end = clock();
    double time_spent = (double)1000*(end - begin) / CLOCKS_PER_SEC;

    //commented out the functions which helps to calculate time
    printf("GPU time= %f ms\n", et);

    printf("CPU time= %lf ms\n", time_spent);

    //Prints the results
    print_matrices(Res_h,"GPU_out",Left_matrix_x,Right_matrix_y,dim);
    print_matrices(CPU,"CPU_out",Left_matrix_x,Right_matrix_y,dim);

    bool eqaul = true;
    for (int i=0;i< Left_matrix_x && eqaul;i++){
        for (int j = 0; j < Right_matrix_y && eqaul; j++) {
            if (abs(Res_h[i*dim+j]-CPU[i*dim+j]) > 0.001)
            {
                eqaul = false;
                printf("NOT EQUAL\n");
            }
        }
    }
    if (eqaul)
    {
        std::cout<<"Results are equal!"<<std::endl;
    }
    else
    {
        std::cout<<"Results are NOT equal!"<<std::endl;
    }

    // Cleanup
    free(Left_Vector_h);
    free(Right_Vector_h);
    free(Res_h);
    free(CPU);
    cudaFree(Left_Vector_d);
    cudaFree(Right_Vector_d);
    cudaFree(Res_d);
}