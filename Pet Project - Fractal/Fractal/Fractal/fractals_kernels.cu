#include "fractals_kernels.cuh"
#include "complex_number.h"

__global__ void GenerateMandelbrotSetKernel(float* pixelMap, const int imageSize, const int height, const int width) {
	unsigned int indx = blockIdx.x * blockDim.x + threadIdx.x;
	// Find x and y coordinates from a linearized array of pixels
	// One needs to remember that we were linearizing 3D array into 1D array
	// So we need to multiply/divide width by 3
	const int y = indx / (width * 3);
	const int x = (indx - y * width * 3) / 3;



	// Find pixels red channel
	if ((indx % 3 == 0) && (indx < imageSize)) {
		const float zoom = 2.2f / static_cast<float>(height);
		const int maxIterations = 200;
		const float xOffset = 1.25f * width / 2.0f;
		const float yOffset = height / 2.0f;

		ComplexCUDA c((x - xOffset) * zoom, (y - yOffset) * zoom);
		ComplexCUDA z(0, 0);
		int iterationNumber = 0;

		for (int i = 0; i < maxIterations; i++) {
			z = z * z + c;
			if (z.Magnitude() > 2.0f) {
				break;
			}
			iterationNumber++;
		}

		// Rewrite pixelMap with yellow colour
		float iteratedColour = 4.0f * iterationNumber / static_cast<float>(maxIterations);
		pixelMap[indx] = iteratedColour;
		pixelMap[indx + 1] = iteratedColour;
	}
}