#ifndef IMAGE_PROCESSING_CUDA_CUH
#define IMAGE_PROCESSING_CUDA_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void MakeGreyScaleKernel(float* pixelMap_in, float* pixelMap_out, const int imageSize,
									float redChannelWeight = 0.299f, float greenChannelWeight = 0.587f, float blueChannelWeight = 0.114f);
__global__ void MakeGreyScaleKernelTexture(float* pixelMap_out, cudaTextureObject_t textureObject, const int width, const int height,
										   float redChannelWeight = 0.299f, float greenChannelWeight = 0.587f, float blueChannelWeight = 0.114f);

__global__ void SobelOperatorKernel(float* pixelMap_in, float* pixelMap_out, const int imageSize, const int height, const int width);
__global__ void SobelOperatorKernelTexture(float* pixelMap_out, cudaTextureObject_t textureObject, const int width, const int height);

#endif // !IMAGE_PROCESSING_CUDA_CUH
