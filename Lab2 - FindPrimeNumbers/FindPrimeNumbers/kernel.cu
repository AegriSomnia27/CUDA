#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <random>
#include <chrono>
#include <limits>
#include <stdio.h>
#include <math.h>

const unsigned int ARRAY_SIZE = 16;
const unsigned int GRID_SIZE = 4;

using PairsVector = std::vector<std::pair<int, std::vector<int>>>; // Используем псевдоним для сложной структуры

// Инициализация генератора случайных чисел, вихря Мерсенна и равномерного распределения с 0 и максимальным значениями
// равными [0, max] значению перменной типа int
std::random_device RandomDevice{};
std::mt19937 Generator(RandomDevice());
std::uniform_int_distribution<int> Distribution(0, std::numeric_limits<int>::max());

__global__ void SharedFindPrimeNumbersGPUKernel(int* NumbersArrayIn,
												int* PrimeNumberArrayOut) {
	// Создаём массив в shared memory
	__shared__ int sdata[4];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = NumbersArrayIn[i];

	__syncthreads();

	// Используем аналогичный алгоритм для GPU, который параллельно вычисляет каждое отдельное число
	for (int j = 2; j <= sqrt(static_cast<double>(sdata[tid])); j++) {
		while (sdata[tid] % j == 0 ) {
			sdata[tid] /= j;
			printf("\nPrime: %i", j);
		}
	}
	if (sdata[tid] != 1) {
		printf("\nLast prime: %i", sdata[tid]);
	}

	__syncthreads();
}

__host__ PairsVector FindPrimeNumbersCPU(int* NumbersArray) {
	// Создаём динамический массив (вектор пар), который состоит из pair: int и vector
	// int - число, простые множители которого необходимо найти
	// vector - массив простых множителей
	PairsVector PrimeNumbersVector(ARRAY_SIZE);

	for (int IndexOfANumber = 0; IndexOfANumber < ARRAY_SIZE; IndexOfANumber++) {
		// Записываем в первый элемент пары число, простые множители которого необходимо найти
		PrimeNumbersVector[IndexOfANumber].first = NumbersArray[IndexOfANumber];

		// Проверяем, делится ли большое число без остатка, если да - записываем результат в вектор пар
		for (int j = 2; j <= sqrt(NumbersArray[IndexOfANumber]); j++) {
			while (NumbersArray[IndexOfANumber] % j == 0) {
				PrimeNumbersVector[IndexOfANumber].second.push_back(j);
				NumbersArray[IndexOfANumber] /= j;
			}
		}
		
		// После того, как прошли весь массив и вышли из цикла - записываем последнее число в вектор пар
		if (NumbersArray[IndexOfANumber] != 1) {
			PrimeNumbersVector[IndexOfANumber].second.push_back(NumbersArray[IndexOfANumber]);
		}
	}

	return PrimeNumbersVector;
}

// Простая обёртка для того, чтобы проверять функции для GPU на ошибки
__host__ inline void CudaErrorHandler(cudaError_t code) {
	if (code != cudaSuccess) {
		std::cout << "There was a fatal error " << cudaGetErrorString(code) << std::endl;
		exit(-1);
	}
}

int main() {
	int ByteSize = ARRAY_SIZE * sizeof(int);
	auto NumbersArrayCPU = new int[ARRAY_SIZE];
	auto NumbersArrayGPU = new int[ARRAY_SIZE];
	auto OutputPrimeNumberArray = new int[ARRAY_SIZE];

	// Генерируем случайные числа с помощью равномерного распределения и закидываем их в два массива для CPU и GPU
	for (int i = 0; i < ARRAY_SIZE; i++) {
		NumbersArrayCPU[i] = Distribution(Generator);
		NumbersArrayGPU[i] = NumbersArrayCPU[i];
	}

	std::cout << "-----------------------Algorithm runs on CPU-----------------------" << std::endl;
	std::cout << "doing computations, please wait..." << std::endl;
	
	// Вычисляем простые множетели на CPU и расчитываем время выполнения
	auto Start = std::chrono::high_resolution_clock::now();
	PairsVector PrimeNumbers = FindPrimeNumbersCPU(NumbersArrayCPU);
	auto Stop = std::chrono::high_resolution_clock::now();

	// Выводим значения простых множителей, полученных на CPU
	for (int i = 0; i < ARRAY_SIZE; i++) {
		std::cout << "Number: " << PrimeNumbers[i].first << "\t\t" << "Prime numbers: ";
		for (int j = 0; j < PrimeNumbers[i].second.size(); j++) {
			std::cout << PrimeNumbers[i].second[j] << ", ";
		}
		std::cout << std::endl;
	}

	// Рассчитываем количество затраченного времени на проведение расчётов на CPU
	auto Duration = std::chrono::duration_cast<std::chrono::microseconds>(Stop - Start);
	std::cout << "The duration of cpu computations is " << Duration.count() << " microseconds" << std::endl << std::endl << std::endl;

	// Использование global memory
	// Выделение памяти для GPU
	int* InputDevNumbersArrayGPU; 
	int* OutputDevNumbersArrayGPU;
	CudaErrorHandler(cudaMalloc(reinterpret_cast<void**>(&InputDevNumbersArrayGPU), ByteSize));
	CudaErrorHandler(cudaMalloc(reinterpret_cast<void**>(&OutputDevNumbersArrayGPU), ByteSize));

	// Копируем память из host'а на device
	CudaErrorHandler(cudaMemcpy(InputDevNumbersArrayGPU, NumbersArrayGPU, ByteSize, cudaMemcpyHostToDevice));

	// Конфигурируем запуск ядра
	dim3 GridSize(GRID_SIZE);
	dim3 BlockSize(ARRAY_SIZE / GRID_SIZE);

	// Переходим к вычислениям на GPU
	std::cout << "------------------Algorithm runs on GPU (Handmade)------------------" << std::endl;
	std::cout << "doing computations, please wait..." << std::endl;

	// Вызываем ядро и проводим вычисления на GPU
	Start = std::chrono::high_resolution_clock::now();
	SharedFindPrimeNumbersGPUKernel <<<GridSize, BlockSize>>> (InputDevNumbersArrayGPU, OutputDevNumbersArrayGPU);
	Stop = std::chrono::high_resolution_clock::now();

	//CudaErrorHandler(cudaMemcpy(OutputPrimeNumberArray, OutputGlobalDevNumbersArrayGPU, ByteSize, cudaMemcpyDeviceToHost));
	//Рассчитываем время проведения вычислений на GPU
	Duration = std::chrono::duration_cast<std::chrono::microseconds>(Stop - Start);
	std::cout << "The duration of Gpu computations is " << Duration.count() << " microseconds" << std::endl << std::endl << std::endl;

	//system("pause");
	return 0;
}