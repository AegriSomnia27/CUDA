#include "image_processing.h"

#include <cmath>
#include <math.h> // for CUDA computations

void ImageProcessing::MakeGreyScale(Bitmap* bmpImage, float redChannelWeight, float greenChannelWeight, float blueChannelWeight){
	for (int y = 0; y < bmpImage->GetImageHeight(); y++) {
		for (int x = 0; x < bmpImage->GetImageWidth(); x++) {
			// Take a current pixel colour
			Colour bitmapColour = bmpImage->GetColour(x, y);
			
			// Calculate grey scale colour and initialize a new colour object with it
			float greyScale = bitmapColour.red * redChannelWeight + bitmapColour.green * greenChannelWeight + bitmapColour.blue * blueChannelWeight;

			// Change pixel's colour
			bmpImage->SetColour(Colour(greyScale, greyScale, greyScale), x, y);
		}
	}
}

void ImageProcessing::SobelOperator(Bitmap* bmpImage){
	// Initializing Sobel Horizontal Mask
	const unsigned int GX[3][3] = {
		{ 1, 0, -1 },
		{ 2, 0, -2 },
		{ 1, 0, -1 }
	};
	
	// Initializing Sobel Vertical Mask
	const unsigned int GY[3][3] = {
		{ 1,  2,  1},
		{ 0,  0,  0},
		{-1, -2, -1}
	};

	for (int i = 0; i < bmpImage->GetImageHeight(); i++) {
		for (int j = 0; j < bmpImage->GetImageWidth(); j++) {
			unsigned char xSum = 0; unsigned char ySum = 0;

			// If the cureent pixel is a part of the image contour initialize xSum and ySum with 0
			// so you would not get outside of the allocated memory
			if ((i == 0) || (j == 0) || (i == bmpImage->GetImageHeight() - 1) || (j == bmpImage->GetImageWidth() - 1)) {
				xSum = 0;
				ySum = 0;
			}
			else {
				for (int y = -1; y <= 1; y++) {
					for (int x = -1; x <= 1; x++) {
						// Calculation are made only for one colour channel, because
						// The image is greysclade, therefore if it's not you should ignore the other two channels
						xSum += bmpImage->GetColour(j + y, i + x).red * GX[y+1][x+1];
						ySum += bmpImage->GetColour(j + y, i + x).red * GY[y+1][x+1];
					}
				}
			}

			// Calculate magnitude
			float sobelValue = std::sqrt(xSum*xSum + ySum*ySum);
			float floatSobelValue = static_cast<float>(sobelValue);

			bmpImage->SetColour(Colour(floatSobelValue, floatSobelValue, floatSobelValue), j, i);

			xSum = 0; ySum = 0;
		}
	}
}
