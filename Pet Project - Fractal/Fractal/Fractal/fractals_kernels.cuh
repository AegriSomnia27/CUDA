#ifndef FRACTALS_KERNEL_CUH
#define FRACTALS_KERNEL_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void GenerateMandelbrotSetKernel(float* pixelMap, const int imageSize, const int height, const int width);

//TO BE IMPLEMENTED
// __global__ void GenerateJuliaSetKernel(float* pixelMap);


#endif // !FRACTALS_KENREL_CUH
