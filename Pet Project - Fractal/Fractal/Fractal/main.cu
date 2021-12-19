#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>

#include "bitmap.h"
#include "complex_number.h"
#include "fractals_generator.h"

int main() {
	const int height = 2160; // 2160
	const int width = 3840; // 3840

	Bitmap bmpCPU(height, width, "Mandelbrot.bmp");
	Bitmap bmpGPU(height, width, "Mandelbrot.bmp");

	auto start = std::chrono::high_resolution_clock::now();
	FractalGenerator::GenerateMandelbrotSetCPU(&bmpCPU);
	auto stop = std::chrono::high_resolution_clock::now();
	bmpCPU.GenerateBitmapImage("Mandelbrot_CPU.bmp");
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "	Computating time on CPU: " << duration.count() << " milliseconds\n\n";

	start = std::chrono::high_resolution_clock::now();
	FractalGenerator::GenerateMandelbrotSetGPU(&bmpGPU);
	stop = std::chrono::high_resolution_clock::now();
	bmpGPU.GenerateBitmapImage("Mandelbrot_GPU.bmp");
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "	Computating time on GPU: " << duration.count() << " milliseconds\n\n";

	return 0;
}