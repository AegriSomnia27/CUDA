#ifndef BISECTION_H
#define BISECTION_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ double Function(double x);
__host__ void BisectionMethod(double a, double b, const double epsilon = 0.01);

__device__ float FunctionCUDA(float x);
__global__ void BisectionMethodCUDA();

#endif 