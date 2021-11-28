#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <iostream>

#include "bitmap.h"


int main() {
	Bitmap bmp("cat.bmp", "testshit.bmp");
	bmp.GenerateBitmapImage();

	return 0;
}