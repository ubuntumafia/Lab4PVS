#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <chrono>

#define ASCENDING 1
#define DESCENDING 0

__device__ void swap(int &a, int &b, bool dir) {
    if ((a > b) == dir) {
        int tmp = a;
        a = b;
        b = tmp;
    }
}

__global__ void bitonicSort(int *arr, int n, int j, int k) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = idx ^ j;

    if (ixj > idx && ixj < n && idx < n) {
        if ((idx & k) == 0) {
            if (arr[idx] > arr[ixj]) {
                int tmp = arr[idx];
                arr[idx] = arr[ixj];
                arr[ixj] = tmp;
            }
        } else {
            if (arr[idx] < arr[ixj]) {
                int tmp = arr[idx];
                arr[idx] = arr[ixj];
                arr[ixj] = tmp;
            }
        }
    }
}

void cudaBitonicSort(int *arr, int size, int threadsPerBlock) {
    int *d_arr;
    cudaMalloc(&d_arr, size * sizeof(int));
    cudaMemcpy(d_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSort<<<blocks, threadsPerBlock>>>(d_arr, size, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Использование: " << argv[0] << " <array_size> <threads_per_block>\n";
        return 1;
    }

    int size = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int *arr = new int[size];
    for (int i = 0; i < size; ++i)
        arr[i] = rand();

    auto start = std::chrono::high_resolution_clock::now();
    cudaBitonicSort(arr, size, threadsPerBlock);
    auto end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Параллельная сортировка (Bitonic Sort) завершена\n";
    std::cout << "Время выполнения: " << duration << " секунд\n";

    delete[] arr;
    return 0;
}
