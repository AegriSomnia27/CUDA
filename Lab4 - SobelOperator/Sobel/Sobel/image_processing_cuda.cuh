#ifndef IMAGE_PROCESSING_CUDA_CUH
#define IMAGE_PROCESSING_CUDA_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// TODO: rewrite this function
// Add them to ImageProcessing class from image_processing.h.
// ImageProcessing invokes a kernel. It's a better implemenation.

__global__ void MakeGreyScaleKernel(float* pixelMap_in, float* pixelMap_out, const int size,
									float redChannelWeight = 0.299f, float greenChannelWeight = 0.587f, float blueChannelWeight = 0.114f);

__global__ void SobelOperatorKernel(float* pixelMap, const int size, const int height, const int width);

#endif // !IMAGE_PROCESSING_CUDA_CUH
