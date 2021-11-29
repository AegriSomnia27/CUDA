#include "image_processing.h"

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
