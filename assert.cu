#include <cuda_runtime_api.h>
//#include <assert.h>

__global__ void simpleKernel(float *input, float *output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx] = input[idx] + 10.0f;
    }
    
    if (idx == 0) {
        output[n] = input[n]; 
    }
}

int main() {
    const int n = 64;
    float *dev_input, *dev_output;
    float input[n], output[n+1]; 

    cudaMalloc((void**)&dev_input, n * sizeof(float));
    cudaMalloc((void**)&dev_output, (n + 1) * sizeof(float)); 

    for (int i = 0; i < n; i++) {
        input[i] = float(i);
    }

    cudaMemcpy(dev_input, input, n * sizeof(float), cudaMemcpyHostToDevice);
    //simpleKernel<<<1, 64>>>(dev_input, dev_output, n);
    cudaDeviceSynchronize();

    cudaMemcpy(output, dev_output, (n + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < n; i++) {
        assert(output[i] == input[i] + 10.0f);
    }

    

    cudaFree(dev_input);
    cudaFree(dev_output);
    return 0;
}
