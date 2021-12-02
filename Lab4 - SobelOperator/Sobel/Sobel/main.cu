#include <iostream>
#include <ctime>
#include <chrono>
//#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "texture_types.h"

#include "bitmap.h"
#include "image_processing.h"
#include "image_processing_cuda.cuh"

const int THREADS = 256;
//const int THREADS = 1;

__host__ inline void CUDAErrorHandler(cudaError_t code) {
	if (code != cudaSuccess) {
		std::cout << "There was a fatal error " << cudaGetErrorString(code) << std::endl;
		exit(-1);
	}
}

int main() {
	// Create an object that represent a .bmp file
	Bitmap bmpCatCPU("cat.bmp", "test.bmp");
	bmpCatCPU.DisplayImageInfo();

	std::cout << "\n\n----------Making computation on the CPU...----------\n";

	// Greyscale an image on CPU
	std::cout << "\n  Greyscaling an image...\n";
	auto start = std::chrono::high_resolution_clock::now();
	ImageProcessing::MakeGreyScale(&bmpCatCPU);
	auto stop = std::chrono::high_resolution_clock::now();

	// Generate a grey scale image
	bmpCatCPU.GenerateBitmapImage("cat_greyscale_CPU.bmp");

	
	auto greyScaleDuration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Duration of greyscaling is " << greyScaleDuration.count() << " milliseconds\n\n";

	// Use Sobel operator for edge detection
	std::cout << "  Using the Sobel operator...\n";
	start = std::chrono::high_resolution_clock::now();
	ImageProcessing::SobelOperator(&bmpCatCPU);
	stop = std::chrono::high_resolution_clock::now();

	// Generate an image after using the sobel operator
	bmpCatCPU.GenerateBitmapImage("cat_sobel_operator_CPU.bmp");

	auto sobelOperatorDuration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Duration of the Sobel operator is " << sobelOperatorDuration.count() << " milliseconds\n\n";

	// Compute total duration on CPU
	std::cout << "Total computation duration on CPU: " << greyScaleDuration.count() + sobelOperatorDuration.count() << " milliseconds\n";
	std::cout << "\n==================================================================================\n\n";

	//---------------------------------------------------------------------------------------------------------------------------------------------
	// Create an object that represent a .bmp file
	Bitmap bmpCatGPU("cat.bmp", "test.bmp");
	bmpCatGPU.DisplayImageInfo();

	std::cout << "\n\n----------Making computation on the GPU...----------\n";

	// Generate raw pixel array for computation on GPU
	float* pixelMap = bmpCatGPU.GenerateLinearizedPixelMap();
	const int BYTES_PER_BMP = bmpCatGPU.GetImageSize() * sizeof(float);

	// Allocate memory on device
	float* devPixelMap_in;
	CUDAErrorHandler(cudaMalloc(reinterpret_cast<void**>(&devPixelMap_in), BYTES_PER_BMP));
	float* devGreyscaledImage;
	CUDAErrorHandler(cudaMalloc(reinterpret_cast<void**>(&devGreyscaledImage), BYTES_PER_BMP));
	float* devSobelImage;
	CUDAErrorHandler(cudaMalloc(reinterpret_cast<void**>(&devSobelImage), BYTES_PER_BMP));
	
	// Generate linearized array and copy it into device memory
	float* linearizedPixelMap = bmpCatGPU.GenerateLinearizedPixelMap();
	CUDAErrorHandler(cudaMemcpy(devPixelMap_in, linearizedPixelMap, BYTES_PER_BMP, cudaMemcpyHostToDevice));

	// Number of blocks for computation
	const int NUMBER_OF_BLOCKS = std::ceil(bmpCatGPU.GetImageSize()/static_cast<float>(THREADS));

	// Greyscale an image on GPU
	std::cout << "\n  Greyscaling an image on " << NUMBER_OF_BLOCKS << " blocks with " << THREADS << " threads...\n";

	cudaEvent_t cudaStart, cudaStop;
	cudaEventCreate(&cudaStart);
	cudaEventCreate(&cudaStop);

	cudaEventRecord(cudaStart);
	MakeGreyScaleKernel << <NUMBER_OF_BLOCKS, THREADS >> > (devPixelMap_in, devGreyscaledImage, bmpCatGPU.GetImageSize());
	cudaEventRecord(cudaStop);

	// Copy pixelMap back to host
	CUDAErrorHandler(cudaMemcpy(pixelMap, devGreyscaledImage, BYTES_PER_BMP, cudaMemcpyDeviceToHost));

	// Generate an image
	bmpCatGPU.InitializeWithLinearizedPixelMap(pixelMap, bmpCatGPU.GetImageHeight(), bmpCatGPU.GetImageWidth()*3);
	bmpCatGPU.GenerateBitmapImage("cat_greyscale_GPU.bmp");

	cudaEventSynchronize(cudaStop);
	float timeGreyScaleCUDA = 0;
	cudaEventElapsedTime(&timeGreyScaleCUDA, cudaStart, cudaStop);
	std::cout << "Duration of greyscaling is " << timeGreyScaleCUDA << " milliseconds\n\n";

	// Copying image from device pointer to device pointer
	CUDAErrorHandler(cudaMemcpy(devPixelMap_in, devGreyscaledImage, BYTES_PER_BMP, cudaMemcpyDeviceToDevice));

	std::cout << "\n  Using the sobel operator on " << NUMBER_OF_BLOCKS << " blocks with " << THREADS << " threads...\n";

	cudaEventRecord(cudaStart);
	SobelOperatorKernel <<<NUMBER_OF_BLOCKS, THREADS >>> (devGreyscaledImage,
															 bmpCatGPU.GetImageSize(), 
															 bmpCatGPU.GetImageHeight(), 
															 bmpCatGPU.GetImageWidth()*3);
	cudaEventRecord(cudaStop);

	// Copy pixelMap back to host
	CUDAErrorHandler(cudaMemcpy(pixelMap, devGreyscaledImage, BYTES_PER_BMP, cudaMemcpyDeviceToHost));

	// Generate an image after using the sobel operator
	bmpCatGPU.InitializeWithLinearizedPixelMap(pixelMap, bmpCatGPU.GetImageHeight(), bmpCatGPU.GetImageWidth() * 3);
	bmpCatGPU.GenerateBitmapImage("cat_sobel_operator_GPU.bmp");


	cudaEventSynchronize(cudaStop);
	float timeSobelOperatorCUDA = 0;
	cudaEventElapsedTime(&timeSobelOperatorCUDA, cudaStart, cudaStop);
	std::cout << "Duration of the Sobel operator is " << timeSobelOperatorCUDA << " milliseconds\n\n";

	// Compute total duration on CPU
	std::cout << "Total computation duration on GPU: " << timeGreyScaleCUDA + timeSobelOperatorCUDA << " milliseconds\n";

	// Free allocated memory
	cudaFree(devPixelMap_in);
	cudaFree(devGreyscaledImage);
	cudaFree(devSobelImage);
	delete[] pixelMap;

	return 0;
}