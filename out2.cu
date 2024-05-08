#include <stdio.h>
#include <assert.h>
#include <cuda_runtime_api.h>

#define BLOCKS 1
#define THREADS 2

__global__ void kernel(int *A) {
    A[threadIdx.x + 1] = threadIdx.x;
}

int main() {
    int *a;
    int *dev_a;
    int size = THREADS * sizeof(int);
    a = (int *)malloc(size);
    cudaMalloc((void **)&dev_a, size);

    for (int i = 0; i < THREADS; i++) {
        a[i] = 0;  // Initialize array with 0
    }

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);

    // Launch kernel
    //kernel<<<BLOCKS, THREADS>>>(dev_a);
    ESBMC_verify_kernel(kernel, BLOCKS, THREADS, dev_a);

    cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < THREADS; i++) {
        assert(a[i] == i);  
    }

    cudaFree(dev_a);
    free(a);
    return 0;
}
