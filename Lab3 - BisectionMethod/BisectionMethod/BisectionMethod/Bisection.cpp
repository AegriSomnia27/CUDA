#include "Bisection.h"

#include <cmath> // дл€ вычислени€ степени на CPU
#include <iostream>
#include <iomanip>

#include <math.h> // дл€ вычислени€ степени на GPU
#include <stdio.h>

__host__ double Function(double x){
	// 0.89x^3 - 2.8x^2 - 3.7x + 11.2 = 0
	return 0.89*std::pow(x,3)-2.8*std::pow(x,2)-3.7*x+11.2;
}

__host__ void BisectionMethod(double leftPoint, double rightPoint, const double epsilon){
	if (Function(leftPoint) * Function(rightPoint) >= 0) {
		//std::cout << "You've assumed wrong a and b\n";
		return;
	}

	double middlePoint = 0.0;
	while ((rightPoint - leftPoint) >= epsilon) {
		// Ќаходим середину заданного отрезка
		middlePoint = (rightPoint + leftPoint) / 2;

		// ≈сли середина - корень уровнени€, то выходим из цикла
		if (Function(middlePoint) == 0.0) {
			break;
		}

		// ¬ыбираем сторону дл€ следующей итерации
		else if (Function(middlePoint) * Function(leftPoint) < 0) {
			rightPoint = middlePoint;
		}
		else {
			leftPoint = middlePoint;
		}
	}

	// ¬ыводим результат в консоль
	std::cout << " The value of the root is: " << std::setprecision(3) << middlePoint << std::endl;
}

__device__ float FunctionCUDA(float x){
	// 0.89x^3 - 2.8x^2 - 3.7x + 11.2 = 0
	return 0.89f * powf(x, 3.0f) - 2.8f * powf(x, 2.0f) - 3.7f * x + 11.2f;
}

__global__ void BisectionMethodCUDA(){
	const float blocks = 65535.0f;
	const float threads = 1024.0f;

	// »нициализируем эпсилон и смещение дл€ того, чтобы получить рабочие интервалы дл€ каждого thread
	const float epsilon = 0.01f;
	const float offset = blocks * threads / 2.0f;
	const float intervalSize = 5.0f;

	// Ќаходим уникальный индекс каждого треда
	unsigned int indx = blockIdx.x * blockDim.x + threadIdx.x;

	// Ќаходим рабочий интервал конкретного thread.
	// ¬се индексы по своей природе - положительные числа. ќдна интервал должен учитывать возможные отрицательные корни.
	// ѕоэтому из каждого индекса вычитаем смещение, которое равно половине от общего числа тредов.
	float leftPoint = static_cast<float>(indx) - offset;
	float rightPoint = static_cast<float>(indx) - offset + intervalSize;


	if (FunctionCUDA(leftPoint) * FunctionCUDA(rightPoint) >= 0) {
		return;
	}

	float middlePoint = 0.0f;
	int iteration = 0;
	const int maxIterations = 100;


	do {
		// Ќаходим середину заданного отрезка
		middlePoint = (rightPoint + leftPoint) / 2;
		

		// ≈сли середина - корень уровнени€, то записываем результат
		if (FunctionCUDA(middlePoint) == 0.0) {
			printf(" The value of the root is: %f\n", middlePoint);
			return;
		}

		// ≈сли разница между левой и правой точкой меньше эпсилон, то выводим корень в консоль
		if (fabsf((rightPoint - leftPoint)) <= epsilon) {
			printf(" The value of the root is: %.2f\n", middlePoint);
			return;
		}

		// ¬ыбираем сторону дл€ следующей итерации
		else if (FunctionCUDA(middlePoint) * FunctionCUDA(leftPoint) < 0) {
			rightPoint = middlePoint;
		}
		else {
			leftPoint = middlePoint;
		}

	} while (iteration < maxIterations);

	// ≈сли привышен максимальное допустимое количество итераций, то считаем, что корн€ на данном интервале не существует
	return;
}