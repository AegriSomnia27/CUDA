#include <iostream>
#include <ctime>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bitmap.h"
#include "image_processing.h"

int main() {
	Bitmap bmp("cat.bmp");

	ImageProcessing::MakeGreyScaleGPU(&bmp, MemoryTypeGPU::TEXTURE);
	bmp.GenerateBitmapImage("cat_greyscale.bmp");

	ImageProcessing::SobelOperatorGPU(&bmp, MemoryTypeGPU::GLOBAL
	);
	//ImageProcessing::SobelOperatorCPU(&bmp);
	bmp.GenerateBitmapImage("cat_sobel.bmp");

	return 0;
}