#include <cudnn.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <assert.h>
//#include <generated_cuda_runtime_api_meta.h>

int main()
{
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t xDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnSetTensor4dDescriptor(
      xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 3, 3);

    cudnnFilterDescriptor_t wDesc;
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnSetFilter4dDescriptor(
      wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 3, 4);

    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(
      convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    cudnnTensorDescriptor_t yDesc;
    cudnnCreateTensorDescriptor(&yDesc);
    int n, c, h, w;
    cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n, &c, &h, &w);
    cudnnSetTensor4dDescriptor(
    yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

    float alpha = 1.0f, beta = 0.0f;
    float* img;
    cudaMalloc((void**)&img, sizeof(float) * 9);  // Allocate memory for the image
    cudaMemcpy(img, (float[]){1, 2, 3, 4, 5, 6, 7, 8, 9}, sizeof(float) * 9, cudaMemcpyHostToDevice);

    float* kernel;
    cudaMalloc((void**)&kernel, sizeof(float) * 4);  // Allocate memory for the kernel
    cudaMemcpy(kernel, (float[]){1, 2, 3, 4}, sizeof(float) * 4, cudaMemcpyHostToDevice);

    float* out;
    cudaMalloc((void**)&out, sizeof(float) * 4);  // Allocate memory for the output


    float* leak;
    cudaMalloc((void**)&leak, sizeof(float) * 100);  

 cudnnConvolutionForward(
    handle,
    &alpha,
    xDesc,
    img,
    wDesc,
    kernel,
    convDesc,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    0,
    0,
    &beta,
    yDesc,
    out,
    1,
    3,
    2,
    1,
    0);

    // Cleanup (intentionally not freeing 'leak')


    cudaFree(img);
    cudaFree(kernel);
    cudaFree(out);

    return 0;
}
