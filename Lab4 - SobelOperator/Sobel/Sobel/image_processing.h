#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include "bitmap.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Computation on CPU
class ImageProcessing {
public:
	ImageProcessing() = delete; // we don't want any instance of the class to be created

	static void					MakeGreyScale(Bitmap* bmpImage, float redChannelWeight = 0.299f,
											  float greenChannelWeight = 0.587f, float blueChannelWeight = 0.114f);

	static void					SobelOperator(Bitmap* bmpImage);
};


// Computation on GPU
class ImageProcessingCUDA {
public:
	ImageProcessingCUDA() = delete; // we don't want any instance of the class to be created
	
	static void __device__		MakeGreyScale(unsigned char* pixelsArray, float redChannelWeight = 0.299f,
											  float greenChannelWeight = 0.587f, float blueChannelWeight = 0.114f);

	static void __device__		SobelOperator(unsigned char* pixelsArray, const int height, const int width);
};

#endif // !IMAGE_PROCESSING_H