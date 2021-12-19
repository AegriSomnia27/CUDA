#ifndef FRACTALS_GENERATOR_H
#define FRACTALS_GENERATOR_H

#include "bitmap.h"

class FractalGenerator {
public:
	FractalGenerator() = delete;

	// Mandelbrot set generators
	static void						GenerateMandelbrotSetCPU(Bitmap* bmpImage);
	static void						GenerateMandelbrotSetGPU(Bitmap* bmpImage);

	// Julia set generators
	// TO BE IMPLEMENTED
	// static void						GenerateJuliaSetCPU(Bitmap* bmpImage);
	// static void						GenerateJuliaSetGPU(Bitmap* bmpImage);
};

#endif // !FRACTALS_GENERATOR_H
