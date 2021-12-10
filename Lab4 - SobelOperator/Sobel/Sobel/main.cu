#include <iostream>
#include <ctime>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bitmap.h"
#include "image_processing.h"

int main() {
	Bitmap bmp("cat.bmp");
	ImageProcessing::MakeGreyScaleCPU(&bmp, 0.2126f, 0.7152f, 0.0722f);
	bmp.GenerateBitmapImage("greyscaled_cat.bmp");
	ImageProcessing::NormalizeImageCPU(&bmp);
	bmp.GenerateBitmapImage("normalized_cat.bmp");
	ImageProcessing::SobelOperatorCPU(&bmp);
	bmp.GenerateBitmapImage("sobel_cat.bmp");


	return 0;
}