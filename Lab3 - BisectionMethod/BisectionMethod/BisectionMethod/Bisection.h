#ifndef BISECTION_H
#define BISECTION_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Функции, исполняемые на CPU
__host__ double Function(double x);
__host__ void BisectionMethod(double leftPoint, double rightPoint, const double epsilon = 0.01);

// Функции, исполняемые на GPU
__device__ float FunctionCUDA(float x);
__global__ void BisectionMethodCUDA();


#endif // !BISECTION_H
