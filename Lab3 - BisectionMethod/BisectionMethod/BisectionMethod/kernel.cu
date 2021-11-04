#include "Bisection.h"
#include <iostream>

// Наибольшая степень заданного уравнения
const int numberOfRoots = 3;

// Константы для инициализации ядра
const int blocks = 65535;
const int threads = 1024;

int main() {
	std::cout << "Finding roots on the cpu...\n";

	// Чиселки для интервалов и для смещения принимались методом академического тыка
	double leftValue = -4.0;
	double rightValue = -1.0;
	const double offset = 4.0;
	
	// Вызываем функцию бисекции несколько раз для того, чтобы вывести на экран все корни
	for (int i = 0; i < numberOfRoots; i++) {
		BisectionMethod(leftValue, rightValue);
		leftValue = rightValue;
		rightValue += offset;
	}

	std::cout << "\n\n\n"
		<< "Finding roots on the gpu...\n";

	BisectionMethodCUDA << <blocks, threads >> >();

	return 0;
}