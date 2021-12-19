#include "fractals_generator.h"
#include "complex_number.h"
#include "fractals_kernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

// Simple CUDA error handler
__host__ inline void CUDAErrorHandler(cudaError_t code) {
	if (code != cudaSuccess) {
		std::cout << "There was a fatal error " << cudaGetErrorString(code) << std::endl;
		exit(-1);
	}
}

void FractalGenerator::GenerateMandelbrotSetCPU(Bitmap* bmpImage){
	const float zoom = 2.2f / static_cast<float>(bmpImage->GetImageHeight());
	const int maxIterations = 200;
	const float xOffset = 1.25 * bmpImage->GetImageWidth() / 2.0f;
	const float yOffset = bmpImage->GetImageHeight() / 2.0f;

	for (int y = 0; y < bmpImage->GetImageHeight(); y++) {
		for (int x = 0; x < bmpImage->GetImageWidth(); x++) {
			Complex c((x - xOffset) * zoom, (y - yOffset) * zoom);
			Complex z(0, 0);
			int iterationNumber = 0;

			for (int i = 0; i < maxIterations; i++) {
				z = z * z + c;
				if (z.Magnitude() > 2.0f) {
					break;
				}
				iterationNumber++;
			}

			float iteratedColour = 4.0f * iterationNumber / static_cast<float>(maxIterations);
			bmpImage->SetColour(Colour(0, iteratedColour, iteratedColour), x, y);
		}
	}
}

void FractalGenerator::GenerateMandelbrotSetGPU(Bitmap* bmpImage){
	// Generate a raw 1D pixels array for computation on GPU
	float* rawLinearizedPixelMap = bmpImage->GenerateLinearizedPixelMap();
	const int BYTES_PER_BMP = bmpImage->GetImageSize() * sizeof(float);

	// Allocate memory
	float* devPixelMap;
	CUDAErrorHandler(cudaMalloc(reinterpret_cast<void**>(&devPixelMap), BYTES_PER_BMP));
	CUDAErrorHandler(cudaMemcpy(devPixelMap, rawLinearizedPixelMap, BYTES_PER_BMP, cudaMemcpyHostToDevice));

	// Number of blocks for computation
	const int THREADS = 256;
	const int BLOCKS = std::ceil(bmpImage->GetImageSize() / static_cast<float>(THREADS));
	//const int imageSize, const int height, const int width
	GenerateMandelbrotSetKernel<<<BLOCKS, THREADS>>> (devPixelMap, 
													  bmpImage->GetImageSize(),
													  bmpImage->GetImageHeight(),
													  bmpImage->GetImageWidth());
	// Copy memory back from GPU to CPU
	CUDAErrorHandler(cudaMemcpy(rawLinearizedPixelMap, devPixelMap, BYTES_PER_BMP, cudaMemcpyDeviceToHost));
	bmpImage->InitializeWithLinearizedPixelMap(rawLinearizedPixelMap, bmpImage->GetImageHeight(), bmpImage->GetImageWidth() * 3); // *3 - bytes per pixel

	// Free allocated memory
	CUDAErrorHandler(cudaFree(devPixelMap));
	delete[] rawLinearizedPixelMap;
}
