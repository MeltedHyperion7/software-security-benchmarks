#include <cublas.h>
#include <cuda_runtime_api.h>


int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Define dimensions for matrix multiplication A * B = C
    int rowsA = 2, colsA = 3;
    int rowsB = 3, colsB = 2;
    int rowsC = rowsA, colsC = colsB;

    // Allocate memory for matrices A, B, and C
    float *A, *B, *C;
    cudaMalloc((void **)&A, rowsA * colsA * sizeof(float));
    cudaMalloc((void **)&B, rowsB * colsB * sizeof(float));
    cudaMalloc((void **)&C, rowsC * colsC * sizeof(float));

    // Initialize matrices A and B
    // Matrix A
    for (int i = 0; i < rowsA * colsA; i++) {
        A[i] = 1.0f;
    }
    // Matrix B
    for (int i = 0; i < rowsB * colsB; i++) {
        B[i] = 2.0f;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform matrix multiplication A * B = C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rowsA, colsB, colsA,
                &alpha, A, rowsA, B, rowsB, &beta, C, rowsA);

    // Synchronize to wait for completion
    cudaDeviceSynchronize();

    // Assert that an incorrect element (e.g., C[0, 1] instead of C[0, 0]) is a specific value
    assert(C[1] == 6.0f);  // This is deliberately incorrect. C[1] should actually be 12.0f

    // Free resources
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    cublasDestroy(handle);

    return 0;
}
