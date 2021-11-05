#include "Bisection.h"
#include <iostream>
#include <chrono>

// Наибольшая степень заданного уравнения
const int numberOfRoots = 3;

// Константы для инициализации ядра
const int blocks = 65535;
const int threads = 1024;

int main() {
	std::cout << "Finding roots on the CPU...\n";

	auto start = std::chrono::high_resolution_clock::now();

	// Чиселки для интервалов и для смещения принимались методом академического тыка
	const double intervalSize = 5.0;
	double leftPoint = -blocks * threads / 2;
	double rightPoint = leftPoint + intervalSize;

	// Вызываем функцию бисекции несколько раз для того, чтобы вывести на экран все корни
	while (rightPoint < blocks * threads / 2) {
		BisectionMethod(leftPoint, rightPoint);
		leftPoint++;
		rightPoint++;
	}

	auto stop = std::chrono::high_resolution_clock::now();

	std::cout << "The CPU time is: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()
		<< " microseconds\n\n"
		<< "Finding roots on the GPU...\n";


	start = std::chrono::high_resolution_clock::now();

	BisectionMethodCUDA << <blocks, threads >> > ();

	stop = std::chrono::high_resolution_clock::now();

	cudaDeviceSynchronize();

	std::cout << "The GPU time is: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()
		<< " microseconds\n\n";

	return 0;
}