#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bitmap.h"
#include "image_processing.h"


int main() {
	Bitmap bmp("cat.bmp", "test.bmp");
	bmp.DisplayImageInfo();

	ImageProcessing::MakeGreyScale(&bmp);
	

	bmp.GenerateBitmapImage();

	return 0;
}