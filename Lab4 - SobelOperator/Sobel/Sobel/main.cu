#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bitmap.h"
#include "image_processing.h"


int main() {
	Bitmap bmpCat("cat.bmp", "test.bmp");
	bmpCat.DisplayImageInfo();

	ImageProcessing::MakeGreyScale(&bmpCat);
	bmpCat.GenerateBitmapImage("cat_greyscale.bmp");

	ImageProcessing::SobelOperator(&bmpCat);
	bmpCat.GenerateBitmapImage("cat_sobel_operator.bmp");

	return 0;
}