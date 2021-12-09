#include "sobel_operator.cuh"

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

__global__ void MakeGreyScaleKernelTexture(float* pixelMap_out, cudaTextureObject_t textureObject, const int width, const int height, 
										   float redChannelWeight, float greenChannelWeight, float blueChannelWeight){
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height && x%3==0) {
		// Ignore all the errors by IntelliSence. We KNOW that we are doing
		float greyscale = tex2D<float>(textureObject, x + 0.5f, y + 0.5f) * redChannelWeight
			+ tex2D<float>(textureObject, x+1+0.5f, y+0.5f) * greenChannelWeight
			+ tex2D<float>(textureObject, x+2+0.5f, y+0.5f) * blueChannelWeight;

		pixelMap_out[y * width + x] = greyscale;
		pixelMap_out[y * width + x + 1] = greyscale;
		pixelMap_out[y * width + x + 2] = greyscale;
	}
}


__global__ void SobelOperatorKernel(float* pixelMap_in, float* pixelMap_out, const int size, const int height, const int width) {
	unsigned int tid = threadIdx.x;
	unsigned int indx = blockIdx.x * blockDim.x + threadIdx.x;

	// Initializing Sobel Horizontal Mask
	const float GX[3][3] = {
		{ 1.0f, 0.0f, -1.0f },
		{ 2.0f, 0.0f, -2.0f },
		{ 1.0f, 0.0f, -1.0f }
	};

	// Initializing Sobel Vertical Mask
	const float GY[3][3] = {
		{ 1.0f,  2.0f,  1.0f},
		{ 0.0f,  0.0f,  0.0f},
		{-1.0f, -2.0f, -1.0f}
	};

	 float xSum = 0;
	 float ySum = 0;

	// Set contour to 0.0.0 colour
	if ((indx % width  == 0) || ((indx+1)%width==0) || (indx < width) || (indx > size - width)) {
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
					xSum += pixelMap_in[indx + y*width + x]*GX[y+1][x+1];
					ySum += pixelMap_in[indx + y*width + x]*GY[y+1][x+1];
				}
			}

			float sobelValue = sqrt(xSum * xSum + ySum * ySum);

			// Fill all channel with the same value
			pixelMap_out[indx] = sobelValue;
			pixelMap_out[indx+1] = sobelValue;
			pixelMap_out[indx+2] = sobelValue;
		}
	}
}

__global__ void SobelOperatorKernelTexture(float* pixelMap_out, cudaTextureObject_t textureObject, const int width, const int height){
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Initializing Sobel Horizontal Mask
	const float GX[3][3] = {
		{ 1.0f, 0.0f, -1.0f },
		{ 2.0f, 0.0f, -2.0f },
		{ 1.0f, 0.0f, -1.0f }
	};

	// Initializing Sobel Vertical Mask
	const float GY[3][3] = {
		{ 1.0f,  2.0f,  1.0f},
		{ 0.0f,  0.0f,  0.0f},
		{-1.0f, -2.0f, -1.0f}
	};

	float xSum = 0;
	float ySum = 0;

	if ((x < width) && (y < height)) {
		if (x % 3 == 0) {
			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					// Ingore all the errors. We KNOW that we are doing
					xSum += tex2D<float>(textureObject, x + i + 0.5, y + j + 0.5) * GX[i + 1][j + 1];
					ySum += tex2D<float>(textureObject, x + i + 0.5, y + j + 0.5) * GY[i + 1][j + 1];
				}
			}

			float sobelValue = sqrt(xSum * xSum + ySum * ySum);

			// Fill all channel with the same value
			pixelMap_out[y * width + x] = sobelValue;
			pixelMap_out[y * width + x + 1] = sobelValue;
			pixelMap_out[y * width + x + 2] = sobelValue;
		}
	}
}



//// NOTE: that was a test function with shared memory it doesn't give any advantage in perfomance
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