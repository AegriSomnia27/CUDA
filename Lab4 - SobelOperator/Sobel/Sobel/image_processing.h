#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H


#include "bitmap.h"

// Choose global or texture memory for GPU computation
enum class MemoryTypeGPU {
	GLOBAL,
	TEXTURE
};

// Image processing class implementation on CPU and GPU
class ImageProcessing {
public:
	ImageProcessing() = delete; // we don't want any instance of the class to be created

	static void					MakeGreyScaleCPU(Bitmap* bmpImage, float redChannelWeight = 0.299f,
												 float greenChannelWeight = 0.587f, float blueChannelWeight = 0.114f);
	static void					NormalizeImageCPU(Bitmap* bmpImage);
	static void					SobelOperatorCPU(Bitmap* bmpImage);

	static void					MakeGreyScaleGPU(Bitmap* bmpImage, MemoryTypeGPU memType, float redChannelWeight = 0.299f,
												 float greenChannelWeight = 0.587f, float blueChannelWeight = 0.114f);
	static void					NormalizeImageGPU(Bitmap* bmpImage);
	static void					SobelOperatorGPU(Bitmap* bmpImage, MemoryTypeGPU memType);
};


#endif // !IMAGE_PROCESSING_H