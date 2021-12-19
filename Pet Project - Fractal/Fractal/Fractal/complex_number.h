#ifndef COMPLEX_NUMBER_H
#define COMPLEX_NUMBER_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Complex {
public:
	Complex();
	Complex(float r, float i);

	float				Magnitude() const;
	Complex				GetConjugatedComplexNumber() const;
	float				GetImaginaryPart() const;
	float				GetRealPart() const;
	void				DisplayNumber() const;

	friend Complex		operator*(const Complex& ComplexLeft, const Complex& ComplexRight);
	friend Complex		operator+(const Complex& ComplexLeft, const Complex& ComplexRight);
	friend Complex		operator-(const Complex& ComplexLeft, const Complex& ComplexRight);
	friend Complex		operator/(const Complex& ComplexLeft, const Complex& ComplexRight);

private:
	float				real;
	float				im;
};

class ComplexCUDA {
public:
	__device__ ComplexCUDA() : real(0), im(0) {}
	__device__ ComplexCUDA(float r, float i) : real(r), im(i) {}

	__device__ float					Magnitude() const;
	__device__ ComplexCUDA				GetConjugatedComplexNumber() const;
	__device__ float					GetImaginaryPart() const;
	__device__ float					GetRealPart() const;
	__device__ void						DisplayNumber() const;

	friend __device__ ComplexCUDA		operator*(const ComplexCUDA& ComplexLeft, const ComplexCUDA& ComplexRight);
	friend __device__ ComplexCUDA		operator+(const ComplexCUDA& ComplexLeft, const ComplexCUDA& ComplexRight);
	friend __device__ ComplexCUDA		operator-(const ComplexCUDA& ComplexLeft, const ComplexCUDA& ComplexRight);
	friend __device__ ComplexCUDA		operator/(const ComplexCUDA& ComplexLeft, const ComplexCUDA& ComplexRight);

private:
	float								real;
	float								im;
};
#endif // !COMPLEX_NUMBER_H
