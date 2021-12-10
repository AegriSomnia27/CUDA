#include "image_processing.h"
#include "sobel_operator.cuh"

#include <cmath>
#include <math.h>
#include <iostream>

// Simple CUDA error handler
__host__ inline void CUDAErrorHandler(cudaError_t code) {
	if (code != cudaSuccess) {
		std::cout << "There was a fatal error " << cudaGetErrorString(code) << std::endl;
		exit(-1);
	}
}

void ImageProcessing::MakeGreyScaleCPU(Bitmap* bmpImage, float redChannelWeight, float greenChannelWeight, float blueChannelWeight){
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

void ImageProcessing::NormalizeImageCPU(Bitmap* bmpImage){
	float maxValue = bmpImage->GetColour(0, 0).red; float minValue = maxValue;

	for (int y = 0; y < bmpImage->GetImageHeight(); y++) {
		for (int x = 0; x < bmpImage->GetImageWidth(); x++) {
			// Check every pixel for max and min value. Assume that we greyscaled an image
			float redChannel = bmpImage->GetColour(x, y).red;
			if (redChannel > maxValue) {
				maxValue = redChannel;
			}
			if (redChannel < minValue) {
				minValue = redChannel;
			}
		}
	}

	float newMax = 0.4f; float newMin = 0.0f;

	for (int y = 0; y < bmpImage->GetImageHeight(); y++) {
		for (int x = 0; x < bmpImage->GetImageWidth(); x++) {
			float redChannel = bmpImage->GetColour(x, y).red;
			float normalizedChannel = (redChannel - minValue) * (newMax - newMin) / (maxValue - minValue) + newMin;
			bmpImage->SetColour(Colour(normalizedChannel, normalizedChannel, normalizedChannel), x, y);
		}
	}
}

void ImageProcessing::SobelOperatorCPU(Bitmap* bmpImage){
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

	Bitmap temp(bmpImage);

	for (int i = 0; i < bmpImage->GetImageHeight(); i++) {
		for (int j = 0; j < bmpImage->GetImageWidth(); j++) {
			float xSum = 0.0f; float ySum = 0.0f;

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
						xSum += temp.GetColour(j + y, i + x).red * GX[y+1][x+1];
						ySum += temp.GetColour(j + y, i + x).red * GY[y+1][x+1];
					}
				}
			}

			// Calculate magnitude
			float sobelValue = sqrt(xSum*xSum + ySum*ySum);

			bmpImage->SetColour(Colour(sobelValue, sobelValue, sobelValue), j, i);

			xSum = 0; ySum = 0;
		}
	}
}

void ImageProcessing::MakeGreyScaleGPU(Bitmap* bmpImage, MemoryTypeGPU memoryType,
									   float redChannelWeight, float greenChannelWeight, float blueChannelWeight){
	// Generate a raw 1D pixels array for computation on GPU
	float* rawLinearizedPixelMap = bmpImage->GenerateLinearizedPixelMap();
	const int BYTES_PER_BMP = bmpImage->GetImageSize() * sizeof(float);

	if (memoryType == MemoryTypeGPU::GLOBAL) {
		// Allocate memory
		float* devPixelMap_in;
		CUDAErrorHandler(cudaMalloc(reinterpret_cast<void**>(&devPixelMap_in), BYTES_PER_BMP));
		float* devGreyscaledImage;
		CUDAErrorHandler(cudaMalloc(reinterpret_cast<void**>(&devGreyscaledImage), BYTES_PER_BMP));

		// Copy memory from CPU to GPU
		CUDAErrorHandler(cudaMemcpy(devPixelMap_in, rawLinearizedPixelMap, BYTES_PER_BMP, cudaMemcpyHostToDevice));

		// Number of blocks for computation
		const int THREADS = 256;
		const int NUMBER_OF_BLOCKS = std::ceil(bmpImage->GetImageSize() / static_cast<float>(THREADS));
		MakeGreyScaleKernel<<<NUMBER_OF_BLOCKS, THREADS>>>(devPixelMap_in, devGreyscaledImage, bmpImage->GetImageSize());

		// Copy memory back from GPU to CPU
		CUDAErrorHandler(cudaMemcpy(rawLinearizedPixelMap, devGreyscaledImage, BYTES_PER_BMP, cudaMemcpyDeviceToHost));
		bmpImage->InitializeWithLinearizedPixelMap(rawLinearizedPixelMap, bmpImage->GetImageHeight(), bmpImage->GetImageWidth() * 3); // *3 - bytes per pixel

		// Free allocated memory
		CUDAErrorHandler(cudaFree(devPixelMap_in));
		CUDAErrorHandler(cudaFree(devGreyscaledImage));
		delete[] rawLinearizedPixelMap;
	}
	else if (memoryType == MemoryTypeGPU::TEXTURE) {
		// Allocate CUDA array in device memory
		cudaChannelFormatDesc channelDesc =
			cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		cudaArray_t cuArray;
		CUDAErrorHandler(cudaMallocArray(&cuArray, &channelDesc, bmpImage->GetImageWidth() * 3, bmpImage->GetImageHeight()));

		// Set pitch of the source (the width in memory on bytes of the 2D array pointed
		// to by source, including padding), we don't have any padding
		const size_t spitch = bmpImage->GetImageWidth() * 3 * sizeof(float);
		// Copy data located at adress rawLinearizedPixelMap in host memory to device memory
		CUDAErrorHandler(cudaMemcpy2DToArray(cuArray, 0, 0, rawLinearizedPixelMap, spitch, bmpImage->GetImageWidth() * 3 * sizeof(float),
											 bmpImage->GetImageHeight(), cudaMemcpyHostToDevice));

		// Specify texture
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		// Specify texture object parameters
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0; // We do not use normalization here

		// Create texture object
		cudaTextureObject_t textureObject = 0;
		CUDAErrorHandler(cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL));

		// Allocate result of greyscaling in device memory
		float* devGreyscaledImage;
		CUDAErrorHandler(cudaMalloc(&devGreyscaledImage, BYTES_PER_BMP));

		// Invoke kernel
		dim3 threadsPerBlock(16, 16);
		dim3 numberOfBlocks((bmpImage->GetImageWidth() * 3 + threadsPerBlock.x - 1) / threadsPerBlock.x,
							(bmpImage->GetImageHeight() + threadsPerBlock.y - 1) / threadsPerBlock.y);

		MakeGreyScaleKernelTexture<<<numberOfBlocks, threadsPerBlock>>>(devGreyscaledImage, 
																		textureObject, 
																		bmpImage->GetImageWidth() * 3,
																		bmpImage->GetImageHeight());
		// Copy data from device back to host
		CUDAErrorHandler(cudaMemcpy(rawLinearizedPixelMap, devGreyscaledImage, BYTES_PER_BMP, cudaMemcpyDeviceToHost));
		bmpImage->InitializeWithLinearizedPixelMap(rawLinearizedPixelMap, bmpImage->GetImageHeight(), bmpImage->GetImageWidth() * 3);

		// Destroy texture object
		CUDAErrorHandler(cudaDestroyTextureObject(textureObject));

		// Free allocated memory
		CUDAErrorHandler(cudaFreeArray(cuArray));
		CUDAErrorHandler(cudaFree(devGreyscaledImage));
		delete[] rawLinearizedPixelMap;
	}
}

void ImageProcessing::SobelOperatorGPU(Bitmap* bmpImage, MemoryTypeGPU memoryType){
	// Generate a raw 1D pixels array for computation on GPU
	float* rawLinearizedPixelMap = bmpImage->GenerateLinearizedPixelMap();
	const int BYTES_PER_BMP = bmpImage->GetImageSize() * sizeof(float);

	if (memoryType == MemoryTypeGPU::GLOBAL) {
		// Allocate memory
		float* devPixelMap_in;
		CUDAErrorHandler(cudaMalloc(reinterpret_cast<void**>(&devPixelMap_in), BYTES_PER_BMP));
		float* devSobelOperatorImage;
		CUDAErrorHandler(cudaMalloc(reinterpret_cast<void**>(&devSobelOperatorImage), BYTES_PER_BMP));

		// Copy memory from CPU to GPU
		CUDAErrorHandler(cudaMemcpy(devPixelMap_in, rawLinearizedPixelMap, BYTES_PER_BMP, cudaMemcpyHostToDevice));

		// Number of blocks for computation
		const int THREADS = 256;
		const int NUMBER_OF_BLOCKS = std::ceil(bmpImage->GetImageSize() / static_cast<float>(THREADS));
		SobelOperatorKernel << <NUMBER_OF_BLOCKS, THREADS >> > (devPixelMap_in, devSobelOperatorImage,
			bmpImage->GetImageSize(),
			bmpImage->GetImageHeight(),
			bmpImage->GetImageWidth() * 3); // *3 bytes per pixel

// Copy memory back from GPU to CPU
		CUDAErrorHandler(cudaMemcpy(rawLinearizedPixelMap, devSobelOperatorImage, BYTES_PER_BMP, cudaMemcpyDeviceToHost));
		bmpImage->InitializeWithLinearizedPixelMap(rawLinearizedPixelMap, bmpImage->GetImageHeight(), bmpImage->GetImageWidth() * 3); // *3 bytes per pixel

		// Free allocated memory
		delete[] rawLinearizedPixelMap;
		cudaFree(devPixelMap_in);
		cudaFree(devSobelOperatorImage);
	}
	else if (memoryType == MemoryTypeGPU::TEXTURE) {
		// Allocate CUDA array in device memory
		cudaChannelFormatDesc channelDesc =
			cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		cudaArray_t cuArray;
		CUDAErrorHandler(cudaMallocArray(&cuArray, &channelDesc, bmpImage->GetImageWidth() * 3, bmpImage->GetImageHeight()));

		// Set pitch of the source (the width in memory on bytes of the 2D array pointed
		// to by source, including padding), we don't have any padding
		const size_t spitch = bmpImage->GetImageWidth() * 3 * sizeof(float);
		// Copy data located at adress rawLinearizedPixelMap in host memory to device memory
		CUDAErrorHandler(cudaMemcpy2DToArray(cuArray, 0, 0, rawLinearizedPixelMap, spitch, bmpImage->GetImageWidth() * 3 * sizeof(float),
			bmpImage->GetImageHeight(), cudaMemcpyHostToDevice));

		// Specify texture
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		// Specify texture object parameters
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0; // We do not use normalization here

		// Create texture object
		cudaTextureObject_t textureObject = 0;
		CUDAErrorHandler(cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL));

		// Allocate result of greyscaling in device memory
		float* devSobelOperatorImage;
		CUDAErrorHandler(cudaMalloc(&devSobelOperatorImage, BYTES_PER_BMP));

		// Invoke kernel
		dim3 threadsPerBlock(16, 16);
		dim3 numberOfBlocks((bmpImage->GetImageWidth() * 3 + threadsPerBlock.x - 1) / threadsPerBlock.x,
							(bmpImage->GetImageHeight() + threadsPerBlock.y - 1) / threadsPerBlock.y);

		SobelOperatorKernelTexture<<<numberOfBlocks, threadsPerBlock>>>(devSobelOperatorImage,
																		textureObject,
																		bmpImage->GetImageWidth() * 3,
																		bmpImage->GetImageHeight());
		// Copy data from device back to host
		CUDAErrorHandler(cudaMemcpy(rawLinearizedPixelMap, devSobelOperatorImage, BYTES_PER_BMP, cudaMemcpyDeviceToHost));
		bmpImage->InitializeWithLinearizedPixelMap(rawLinearizedPixelMap, bmpImage->GetImageHeight(), bmpImage->GetImageWidth() * 3);

		// Destroy texture object
		CUDAErrorHandler(cudaDestroyTextureObject(textureObject));

		// Free allocated memory
		CUDAErrorHandler(cudaFreeArray(cuArray));
		CUDAErrorHandler(cudaFree(devSobelOperatorImage));
		delete[] rawLinearizedPixelMap;
	}
}
