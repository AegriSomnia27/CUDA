#include "image_processing_cuda.cuh"

#include <math.h>

__global__ void MakeGreyScaleKernel(float* pixelMap_in, float* pixelMap_out, const int imageSize,
									float redChannelWeight, float greenChannelWeight, float blueChannelWeight) {
	unsigned int indx = blockIdx.x * blockDim.x + threadIdx.x;

	if ((indx % 3 == 0) && (indx < imageSize)) {
		float greyScale = pixelMap_in[indx] * redChannelWeight + pixelMap_in[indx + 1] * greenChannelWeight + pixelMap_in[indx + 2] * blueChannelWeight;
		pixelMap_out[indx] = greyScale;
		pixelMap_out[indx + 1] = greyScale;
		pixelMap_out[indx + 2] = greyScale;
	}
}

__global__ void SobelOperatorKernel(float* pixelMap, const int size, const int height, const int width) {
	unsigned int tid = threadIdx.x;
	unsigned int indx = blockIdx.x * blockDim.x + threadIdx.x;


	// Initialize horizontal and vertical masks
	//const unsigned int GX[9] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
	//const unsigned int GY[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
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

	unsigned char xSum = 0;
	unsigned char ySum = 0;

	// Set contour to 0.0.0 colour
	if ((indx < width*3) || (indx > size-width*3) || (indx%size==0) || (indx%size == size-1)) {
		xSum = 0;
		ySum = 0;
	}
	else {
		// We use greyscale image so we need to pick only one colour channel
		// indx%3 == 0 --> red colour channel
		// Otherwise if image is not greyscaled - ignore two other channels
		if (indx % 3 == 0) {
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {
					xSum += pixelMap[indx + y*width + x]*GX[y+1][x+1];
					ySum += pixelMap[indx + y*width + x]*GY[y+1][x+1];
					//ySum += bmpImage->GetColour(j + y, i + x).red * GY[y + 1][x + 1];
				}
			}

			float sobelValue = sqrt(static_cast<float>(xSum * xSum + ySum * ySum));

			// Fill all channel with the same value
			pixelMap[indx] = sobelValue;
			pixelMap[indx+1] = sobelValue;
			pixelMap[indx+2] = sobelValue;

		}
	}
}



//// NOTE: that was a test function with shared memory
//
//__global__ void MakeGreyScaleKernel(float* pixelMap_in, float* pixelMap_out, const int imageSize,
//									float redChannelWeight, float greenChannelWeight, float blueChannelWeight) {
//	// Initialize __shared__ memory 
//	extern __shared__ float sharedPixels[256];
//
//	// Find thread unique index and thread index in a block 
//	unsigned int tid = threadIdx.x;
//	unsigned int indx = blockIdx.x * blockDim.x + threadIdx.x;
//
//	__syncthreads();
//	// We must check if we use 'our' memory
//	if (indx < imageSize) {
//		// Fill sharedPixels[]
//		sharedPixels[tid] = pixelMap_in[indx];
//	}
//	__syncthreads();
//
//	if (indx < imageSize) {
//		// Multiply colour channel to their weights 
//		if ((tid % 3 == 0) && ((tid + 2) < 256)) {
//			pixelMap_out[indx] = sharedPixels[tid] * redChannelWeight + sharedPixels[tid + 1] * greenChannelWeight + sharedPixels[tid + 2] * blueChannelWeight;
//			pixelMap_out[indx + 1] = pixelMap_out[indx];
//			pixelMap_out[indx + 2] = pixelMap_out[indx];
//		}
//		else if ((tid % 3) && (tid + 1 < 256)) {
//			pixelMap_out[indx] = sharedPixels[tid] * redChannelWeight + sharedPixels[tid + 1] * greenChannelWeight + pixelMap_in[indx + 2] * blueChannelWeight;
//			pixelMap_out[indx + 1] = pixelMap_out[indx];
//			pixelMap_out[indx + 2] = pixelMap_out[indx];
//		}
//		else {
//			pixelMap_out[indx] = sharedPixels[tid] * redChannelWeight + pixelMap_in[indx + 1] * greenChannelWeight + pixelMap_in[indx + 2] * blueChannelWeight;
//			pixelMap_out[indx + 1] = pixelMap_out[indx];
//			pixelMap_out[indx + 2] = pixelMap_out[indx];
//		}
//	}
//
//	// NOTE: you must use __syncthreads() outside of if-statements,
//	// otherwise you will enter an endless loop
//	__syncthreads();
//}