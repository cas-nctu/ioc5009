/* Includes, system */
#include <cstdio>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <cstdio>
#include <stdio.h>
/* Includes, cuda */
#include <cuda_runtime.h>
#include <cudnn.h>

struct Tensor4d
{
    cudnnTensorDescriptor_t desc;
    void *data;
    size_t data_size;

    Tensor4d(int n, int c, int h, int w)
    {
	cudnnCreateTensorDescriptor(&desc);
	cudnnSetTensor4dDescriptor(desc,
				   CUDNN_TENSOR_NCHW,
				   CUDNN_DATA_FLOAT,
				   n, c, h, w);
	data_size = n * c * h * w;
	cudaMalloc((void**)&data,  data_size * sizeof(float));
    }
    ~Tensor4d()
    {
	cudaFree(data);
    }
};


struct Filter4d
{
    cudnnFilterDescriptor_t desc;
    void *data;
    size_t data_size;

    Filter4d(int n, int c, int h, int w)
    {
	cudnnCreateFilterDescriptor(&desc);
	cudnnSetFilter4dDescriptor(desc,
				   CUDNN_DATA_FLOAT,
				   CUDNN_TENSOR_NCHW,
				   n, c, h, w);
	data_size = n * c * h * w;
	cudaMalloc((void**)&data,  data_size * sizeof(float));
    }
    ~Filter4d()
    {
	cudaFree(data);
    }
};

struct zeros
{
    void *data;
    size_t data_size;
    zeros(std::vector<int>dims)
    {
	data_size = std::accumulate(dims.begin(),
				    dims.end(),
				    1,
				    std::multiplies<int>());
	std::vector<float> host_data(data_size);
	for(int i = 0; i < data_size; i++)
	    host_data[i] = 0;
	
	cudaMalloc((void**)&data,  data_size * sizeof(float));
	cudaMemcpy(data, host_data.data(), data_size * sizeof(float), 
		   cudaMemcpyHostToDevice);
    }
    ~zeros()
    {
	cudaFree(data);
    }
};

int main()
{

    // Filter parameters
    int k, c, r, s;
    // Input parameters
    int n, w, h;
    // padding
    int pad_w, pad_h;
    // stride
    int wstride, hstride;
    // init conv layer parameter
    w = 224;
    h = 224;
    c = 3;
    n = 2;
    k = 64;
    r = 7;
    s = 7;
    pad_w = 3;
    pad_h = 3;
    wstride = 2;
    hstride = 2;

    size_t fwd_workspace_size;
    cudnnConvolutionFwdAlgo_t fwd_algo;

    const float alpha = 1.f;
    const float beta = 0.f;

    cudnnHandle_t cudnn_handle;
    // create cudnn handle
    cudnnCreate(&cudnn_handle);

    // datatype
    cudnnDataType_t dataType;
    dataType = CUDNN_DATA_FLOAT;

    // convolution mode
    cudnnConvolutionMode_t mode;
    mode = CUDNN_CONVOLUTION;

    int out_h, out_w, out_c, out_n;
    std::vector<int> output_dims_;

    // create conv desc
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor( conv_desc,
    				     pad_h,
    				     pad_w,
    				     hstride,
    				     wstride,
    				     r,
    				     s,
    				     mode,
    				     dataType);

    // tensor desc
    Tensor4d x_desc(n, c, h, w);

    // filter desc
    Filter4d w_desc(k, c, r, s);

    // get conv dim    
    cudnnGetConvolution2dForwardOutputDim( conv_desc,
    					   x_desc.desc,
    					   w_desc.desc,
    					   &out_n,
    					   &out_c,
    					   &out_h,
    					   &out_w);

    Tensor4d h_desc(out_n, out_c, out_h, out_w);

    output_dims_ = {out_w, out_h, out_c, out_n};

    // choose forward algorith
    const int requestAlgoCount = 1;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults;

    cudnnFindConvolutionForwardAlgorithm( cudnn_handle,
    					  x_desc.desc,
    					  w_desc.desc,
    					  conv_desc,
    					  h_desc.desc,
    					  requestAlgoCount,
    					  &returnedAlgoCount,
    					  &perfResults);
    fwd_algo = perfResults.algo;

    // get workspace size
    cudnnGetConvolutionForwardWorkspaceSize( cudnn_handle,
    					     x_desc.desc,
    					     w_desc.desc,
    					     conv_desc,
    					     h_desc.desc,
    					     fwd_algo,
    					     &fwd_workspace_size);
    
    std::vector<int> u = std::vector<int>{static_cast<int>
				(fwd_workspace_size / sizeof(float)), 1};
   
   // init workspace
    zeros fwd_workspace(u);

    auto start = std::chrono::steady_clock::now();
   // fwd conv


    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration< double,
				    std::micro>(end - start).count());

    std::cout << "conv fwd time:" << fwd_time << " ms" << std::endl;

    // destroy conv desc
    cudnnDestroyConvolutionDescriptor(conv_desc);
    
    return 0;
}
