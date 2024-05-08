#include <cuda_runtime_api.h>
//#include <assert.h>

__global__ void simpleKernel(int *input, int *output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx] = input[idx] + 10.0;
    }
    
    if (idx == 0) {
        output[n] = input[n]; 
    }
}

int main() {
    const int n = 64;
    int *dev_input, *dev_output;
    int input[n], output[n+1]; 

    cudaMalloc((void**)&dev_input, n * sizeof(int));
    cudaMalloc((void**)&dev_output, (n + 1) * sizeof(int)); 

    for (int i = 0; i < n; i++) {
        input[i] = int(i);
    }

    cudaMemcpy(dev_input, input, n * sizeof(int), cudaMemcpyHostToDevice);
    //simpleKernel<<<1, 64>>>(dev_input, dev_output, n);
    cudaDeviceSynchronize();

    cudaMemcpy(output, dev_output, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    return 0;
}
