#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include "bitmap.h"

class ImageProcessing {
public:
	ImageProcessing() = delete;

	static void		MakeGreyScale(Bitmap* bmpImage,float redChannelWeight = 0.299f,
								  float greenChannelWeight = 0.587f, float blueChannelWeight = 0.114f);

	static void		SobelOperator(Bitmap* bmpImage);
};

#endif // !IMAGE_PROCESSING_H