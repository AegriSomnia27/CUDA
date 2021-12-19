#include "complex_number.h"

#include <iostream>
#include <cmath>
#include <math.h>

//----------Complex class on CPU----------

Complex::Complex(): real(0.0f), im(1.0f) {}

Complex::Complex(float r, float i): real(r), im(i) {}

float Complex::Magnitude() const{
	return std::sqrtf(real * real + im * im);
}

Complex Complex::GetConjugatedComplexNumber() const{
	return Complex(real, -im);
}

float Complex::GetImaginaryPart() const{
	return im;
}

float Complex::GetRealPart() const{
	return real;
}

void Complex::DisplayNumber() const{
	std::cout << '(' << real << "; " << im << ')\n';
}

Complex operator*(const Complex& ComplexLeft, const Complex& ComplexRight) {
	return Complex(ComplexLeft.real * ComplexRight.real - ComplexLeft.im * ComplexRight.im,
				   ComplexLeft.real * ComplexRight.im + ComplexLeft.im * ComplexRight.real);
}

Complex operator+(const Complex& ComplexLeft, const Complex& ComplexRight) {
	return Complex(ComplexLeft.real + ComplexRight.real,
				   ComplexLeft.im + ComplexRight.im);
}

Complex operator-(const Complex& ComplexLeft, const Complex& ComplexRight) {
	return Complex(ComplexLeft.real - ComplexRight.real,
				   ComplexLeft.im - ComplexRight.im);
}

Complex operator/(const Complex& ComplexLeft, const Complex& ComplexRight) {
	Complex numerator = ComplexLeft * ComplexRight.GetConjugatedComplexNumber();
	float denominator = ComplexRight.real * ComplexRight.real + ComplexRight.im * ComplexRight.im;

	if (ComplexLeft.real == 0 && ComplexRight.real == 0 && ComplexLeft.im == 0 && ComplexRight.im == 0) {
		throw std::invalid_argument("Bad complex numbers");
	}

	return Complex(numerator.real / denominator, numerator.im / denominator);
}

//----------Complex class on GPU----------

__device__ float ComplexCUDA::Magnitude() const{
	return sqrt(real * real + im * im);
}

__device__ ComplexCUDA ComplexCUDA::GetConjugatedComplexNumber() const{
	return ComplexCUDA(real, -im);
}

__device__ float ComplexCUDA::GetImaginaryPart() const{
	return im;
}

__device__ float ComplexCUDA::GetRealPart() const{
	return real;
}

__device__ void ComplexCUDA::DisplayNumber() const{
	printf("(%f; %f)\n", real, im);
}

__device__ ComplexCUDA operator*(const ComplexCUDA& ComplexLeft, const ComplexCUDA& ComplexRight) {
	return ComplexCUDA(ComplexLeft.real * ComplexRight.real - ComplexLeft.im * ComplexRight.im,
		ComplexLeft.real * ComplexRight.im + ComplexLeft.im * ComplexRight.real);
}

__device__ ComplexCUDA operator+(const ComplexCUDA& ComplexLeft, const ComplexCUDA& ComplexRight) {
	return ComplexCUDA(ComplexLeft.real + ComplexRight.real,
		ComplexLeft.im + ComplexRight.im);
}

__device__ ComplexCUDA operator-(const ComplexCUDA& ComplexLeft, const ComplexCUDA& ComplexRight) {
	return ComplexCUDA(ComplexLeft.real - ComplexRight.real,
		ComplexLeft.im - ComplexRight.im);
}

__device__ ComplexCUDA operator/(const ComplexCUDA& ComplexLeft, const ComplexCUDA& ComplexRight) {
	ComplexCUDA numerator = ComplexLeft * ComplexRight.GetConjugatedComplexNumber();
	float denominator = ComplexRight.real * ComplexRight.real + ComplexRight.im * ComplexRight.im;

	return ComplexCUDA(numerator.real / denominator, numerator.im / denominator);
}