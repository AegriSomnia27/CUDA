#include <iostream>
#include <ctime>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bitmap.h"
#include "image_processing.h"

int main() {
	Bitmap bmpGPUtexture("cat.bmp");
	Bitmap bmpGPUglob("cat.bmp");
	Bitmap bmpCPU("cat.bmp");
	std::cout << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	ImageProcessing::MakeGreyScaleGPU(&bmpGPUtexture, MemoryTypeGPU::TEXTURE);
	auto stop = std::chrono::high_resolution_clock::now();
	auto durationGS = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "GPU texture memory (greyscale): " << durationGS.count() << " microseconds\n";


	start = std::chrono::high_resolution_clock::now();
	ImageProcessing::SobelOperatorGPU(&bmpGPUtexture, MemoryTypeGPU::TEXTURE);
	stop = std::chrono::high_resolution_clock::now();
	auto durationSob = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "GPU texture memory (sobel): " << durationSob.count() << " microseconds\n";
	std::cout << "Total time texture memory: " << durationGS.count() + durationSob.count() << " microseconds\n\n\n";
	


	start = std::chrono::high_resolution_clock::now();
	ImageProcessing::MakeGreyScaleGPU(&bmpGPUglob, MemoryTypeGPU::GLOBAL);
	stop = std::chrono::high_resolution_clock::now();
	durationGS = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "GPU global memory (greyscale): " << durationGS.count() << " microseconds\n";

	start = std::chrono::high_resolution_clock::now();
	ImageProcessing::SobelOperatorGPU(&bmpGPUglob, MemoryTypeGPU::GLOBAL);
	stop = std::chrono::high_resolution_clock::now();
	durationSob = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "GPU global memory (sobel): " << durationSob.count() << " microseconds\n";
	std::cout << "Total time global memory: " << durationGS.count() + durationSob.count() << " microseconds\n\n\n";



	start = std::chrono::high_resolution_clock::now();
	ImageProcessing::MakeGreyScaleCPU(&bmpCPU);
	stop = std::chrono::high_resolution_clock::now();
	durationGS = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "CPU (greyscale): " << durationGS.count() << " microseconds\n";

	start = std::chrono::high_resolution_clock::now();
	ImageProcessing::SobelOperatorCPU(&bmpGPUglob);
	stop = std::chrono::high_resolution_clock::now();
	durationSob = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "CPU (sobel): " << durationSob.count() << " microseconds\n";
	std::cout << "Total time: " << durationGS.count() + durationSob.count() << " microseconds\n\n\n";



	return 0;
}