// Алгоритм редукции с измерением скорости выполнения программы на :
// CPU (последовательное выполнение), GPU (параллельное выполнение)
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <random>
#include <chrono>
#include <limits>

const int ARRAY_SIZE = 16777216; // 32, 64, 128, 256, 512, 1024 ... 16777216
const int GRID_SIZE = 1;

// Инициализация генератора случайных чисел, вихря Мерсенна и равномерного распределения с минимальным и максимальным значениями
// равными минимальному/максимальному значению перменной типа int
std::random_device randomDevice{};
std::mt19937 gen(randomDevice());
std::uniform_int_distribution<int> dist(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());


__host__ int findTheMaxElementCPU(int* arrayCPU) { // Вычисление максимального элемента массива на host'e в CPU
	int theMaxElement = arrayCPU[0];

	for (int i = 1; i < ARRAY_SIZE; i++) {
		if (arrayCPU[i] > theMaxElement)
			theMaxElement = arrayCPU[i];
	}

	return theMaxElement;
}

__global__ void reduction(int* arrayGPU_in, int* arrayGPU_out) {
	for (int i = ARRAY_SIZE; i < 1024; i /= 1024) {
		dim3 gridSize(i/1024);
		dim3 blockSize(ARRAY_SIZE / gridSize.x); // в итоге получаем 1024
		findTheMaxElementGPU <<<gridSize, blockSize>>> (dev_arrayGPU_in, dev_arrayGPU_out);
	}
}
__device__ void findTheMaxElementGPU(int* arrayGPU_in, int* arrayGPU_out) {				// Вычисление максимального элемента в block'е на GPU
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;

	extern __shared__ int shrArrayGPU[];

	for (unsigned int j = 1; j < blockDim.x; j *= 2) {
		if (i % (2 * j) == 0) {
			if (arrayGPU_in[i] < arrayGPU_in[i + j]) {
				arrayGPU_in[i] = arrayGPU_in[i + j];
			}
		}
	}
	if (tid == 0) arrayGPU_out[blockIdx.x] = arrayGPU_in[tid];
}

__host__ inline void cudaErrorHandler(cudaError_t code) {
	if (code != cudaSuccess) {
		std::cout << "There was a fatal error " << cudaGetErrorString(code) << std::endl;
		exit(-1);
	}
}

int main() {
	std::cout << "The current size of an array (N) is " << ARRAY_SIZE << std::endl << std::endl;

	// Выделяем память из кучи для создания массива, содержащего N чисел типа int
	int byteSize = ARRAY_SIZE * sizeof(int);
	auto arrayCPU = new int[ARRAY_SIZE]; // Массив для вычисления на CPU
	auto arrayGPU = new int[ARRAY_SIZE]; // Массив для вычисления на GPU

	// Инициализируем первый массив и приравниваем значение остальных массивов к первому (для адекватности результатов вычисления)
	for (unsigned int i = 0; i < ARRAY_SIZE; i++) {
		arrayCPU[i] = dist(gen);
		arrayGPU[i] = arrayCPU[i];
	}

	// Часть, выполняемая на host'е с расчётом времени выполнения функции
	std::cout << "-----------------------Algorithm runs on CPU-----------------------" << std::endl;
	std::cout << "doing computations, please wait..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	std::cout << "The maximum element of the array is " << findTheMaxElementCPU(arrayCPU) << std::endl;
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "The duration of cpu computations is " << duration.count() << " microseconds" << std::endl << std::endl << std::endl;

	// Выделение памяти для GPU
	int* dev_arrayGPU_in;
	cudaErrorHandler(cudaMalloc(reinterpret_cast<void**>(&dev_arrayGPU_in), byteSize));
	int* dev_arrayGPU_out;
	cudaErrorHandler(cudaMalloc(reinterpret_cast<void**>(&dev_arrayGPU_out), byteSize));

	// Копируем память из host'а на device
	cudaErrorHandler(cudaMemcpy(dev_arrayGPU_in, arrayGPU, byteSize, cudaMemcpyHostToDevice));

	// Конфигурируем запуск ядра
	//dim3 gridSize(GRID_SIZE);
	//dim3 blockSize(ARRAY_SIZE / GRID_SIZE); // Должно получится 1024 - максимальное значение для моего GPU 

	// Вызываем ядро и проводим вычисления на GPU
	start = std::chrono::high_resolution_clock::now();
	reduction <<<1,1>>> (dev_arrayGPU_in, dev_arrayGPU_out); // Вызыв ядра с заданными параметрами 
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	// Копирование результатов вычисления из Device в Host
	cudaErrorHandler(cudaMemcpy(arrayGPU, dev_arrayGPU_out, byteSize, cudaMemcpyDeviceToHost));
	std::cout << "------------------Algorithm runs on GPU (Handmade)------------------" << std::endl;
	std::cout << "doing computations, please wait..." << std::endl;
	std::cout << "The maximum element of the array is " << arrayGPU[0] << std::endl;
	std::cout << "The duration of gpu computations (handmade algorithm) is " << duration.count() << " microseconds" << std::endl << std::endl;

	// Освобождаем выделенную память для host'a и для device
	delete[] arrayCPU;
	delete[] arrayGPU;
	cudaErrorHandler(cudaFree(dev_arrayGPU_out));

	return 0;
}