#include <iostream>
#include <cuda_runtime.h>

__global__ void elementwise_ops(const float* A, const float* B, float* add, float* sub, float* mul, float* div, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        add[idx] = A[idx] + B[idx];
        sub[idx] = A[idx] - B[idx];
        mul[idx] = A[idx] * B[idx];
        div[idx] = A[idx] / B[idx];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Использование: " << argv[0] << " <array_size> <threads_per_block>\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    int threadsPerBlock = std::stoi(argv[2]);
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    size_t size = n * sizeof(float);

    float *A, *B, *add, *sub, *mul, *div;
    float *d_A, *d_B, *d_add, *d_sub, *d_mul, *d_div;

    A = new float[n]; B = new float[n];
    add = new float[n]; sub = new float[n]; mul = new float[n]; div = new float[n];

    for (int i = 0; i < n; ++i) {
        A[i] = 1.5f;
        B[i] = 2.5f;
    }

    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size);
    cudaMalloc(&d_add, size); cudaMalloc(&d_sub, size);
    cudaMalloc(&d_mul, size); cudaMalloc(&d_div, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    elementwise_ops<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_add, d_sub, d_mul, d_div, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, stop, start);

    std::cout << "Время выполнения CUDA: " << ms / 1000.0 << " секунд\n";

    cudaMemcpy(add, d_add, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(sub, d_sub, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(mul, d_mul, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(div, d_div, size, cudaMemcpyDeviceToHost);

    delete[] A; delete[] B; delete[] add; delete[] sub; delete[] mul; delete[] div;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_add); cudaFree(d_sub); cudaFree(d_mul); cudaFree(d_div);

    return 0;
}
