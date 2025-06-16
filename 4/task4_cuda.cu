#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

__global__ void array_ops_2d_kernel(
    float* A, float* B, float* C, 
    size_t rows, size_t cols, int op
) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        const size_t idx = row * cols + col;
        switch(op) {
            case 0: C[idx] = A[idx] + B[idx]; break;
            case 1: C[idx] = A[idx] - B[idx]; break;
            case 2: C[idx] = A[idx] * B[idx]; break;
            case 3: C[idx] = A[idx] / B[idx]; break;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <rows> <cols> <operation> <block_size>\n";
        cerr << "Operations: 0-add, 1-sub, 2-mul, 3-div\n";
        return 1;
    }

    const size_t rows = stoul(argv[1]);
    const size_t cols = stoul(argv[2]);
    const int op = stoi(argv[3]);
    const int block_size = stoi(argv[4]);
    const size_t N = rows * cols;
    
    if (N < 100000) {
        cerr << "Error: Total elements must be >= 100000\n";
        return 1;
    }

    // Выделение памяти на хосте
    vector<float> h_A(N), h_B(N), h_C(N);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const size_t idx = i * cols + j;
            h_A[idx] = (i * cols + j + 1) * 0.1f;
            h_B[idx] = (N - i * cols - j) * 0.1f;
            if (op == 3 && fabs(h_B[idx]) < 1e-6f) h_B[idx] = 1.0f;
        }
    }

    // Выделение памяти на устройстве
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    
    // Копирование данных на устройство
    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Конфигурация запуска ядра
    dim3 threadsPerBlock(block_size, block_size);
    dim3 blocksPerGrid(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    auto t0 = chrono::high_resolution_clock::now();
    
    // Запуск ядра
    array_ops_2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols, op);
    cudaDeviceSynchronize();
    
    // Копирование результата обратно
    cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    auto t1 = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration_cast<chrono::duration<double>>(t1 - t0).count();
    
    // Проверка ошибок CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl;
    }

    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cout << elapsed << "\n";
    return 0;
}