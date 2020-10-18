/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
/* Matrix size */
#define N  (128)
/* Main */
int main(int argc, char **argv)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    float *h_A;
    float *h_B;
    float *h_C;
    float *d_A;
    float *d_B;
    float *d_C;
    long n2 = N * N;    
    cublasHandle_t handle;
    /* Initialize CUBLAS */
    printf("simpleCUBLAS test running..\n");
    
    cublasCreate(&handle);
    /* Allocate host memory for the matrices */
    h_A = (float*)malloc(n2 * sizeof(float));
    h_B = (float*)malloc(n2 * sizeof(float));
    h_C = (float*)malloc(n2 * sizeof(float));
    /* Fill the matrices with test data */
    for (int i = 0; i < n2; i++)
    {   
        h_A[i] = 2.0;
        h_B[i] = 3.0;
        h_C[i] = 0;
    }
    cudaMalloc((void **)&d_A, n2 * sizeof(float));
    cudaMalloc((void **)&d_B, n2 * sizeof(float));
    cudaMalloc((void **)&d_C, n2 * sizeof(float));

    cudaMemcpy(d_A, h_A, n2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, n2 * sizeof(float), cudaMemcpyHostToDevice);

    //cublas kernel
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    
    //free memory   
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cublasDestroy(handle);
}

