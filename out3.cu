
#include <cublas.h>
#include <cuda_runtime_api.h>

int main() {
    cublasHandle_t handle;
    float n = 2;
    float *A, *B, *C;
    float *dev_A, *dev_B, *dev_C;
    // Initialising the cuBLAS environment
    cublasCreate(&handle);
    // Allocating Locked Page Host Memory
    cudaMalloc((void **)&A, 2 * 2 * sizeof(float));
    cudaMalloc((void **)&B, 3 * 3 * sizeof(float));
    cudaMalloc((void **)&C, 3 * 3 * sizeof(float));
    for (int i = 0; i < 2 * 2; i++) {
        A[i] = 1.0f;
    }
    for (int i = 0; i < 3 * 3; i++) {
        B[i] = 1.0f;
    }   
    cudaMalloc((void **)&dev_A, n * n * sizeof(float));
    cudaMalloc((void **)&dev_B, n * n * sizeof(float));
    cudaMalloc((void **)&dev_C, n * n * sizeof(float));
    // Copying data to device
    cudaMemcpy(dev_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, n * n * sizeof(float), cudaMemcpyHostToDevice);
    float alpha = 1.0f;
    float beta = 0.0f;
    // C = A * B
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dev_A, 2, dev_B, 3, &beta, dev_C, 3);
    // Copy results back to host
    cudaMemcpy(C, dev_C, 4 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4 * 4; i++) {
        assert(C[i] == 4.0f);  
    }
    // Liquidation of resources
    cublasDestroy(handle);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}
