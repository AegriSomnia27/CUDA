#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bitmap.h"


int main() {
	Bitmap bmp("cat.bmp", "test.bmp");
	bmp.DisplayImageInfo();
	bmp.GenerateBitmapImage();

	return 0;
}