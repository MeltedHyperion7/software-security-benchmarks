
#include <cuda_runtime_api.h>

__global__ void buggyKernel(int *array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {

        array[idx] = idx * 10;
    }


    if (idx == 0) {
        array[size] = 999; 
    }
}

int main() {
    const int size = 64;
    int *dev_array;

    cudaMalloc((void**)&dev_array, size * sizeof(int));


    //buggyKernel<<<1, 64>>>(dev_array, size);
    cudaDeviceSynchronize();

    int *host_array = (int *)malloc((size + 1) * sizeof(int)); 
    cudaMemcpy(host_array, dev_array, (size + 1) * sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(dev_array);
    free(host_array);
    return 0;
}
