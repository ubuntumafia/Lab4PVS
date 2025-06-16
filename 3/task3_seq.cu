#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

// CPU: арифметические операции
void cpu_arith(const float* A, const float* B, float* C, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
        C[i] = A[i] - B[i];
        C[i] = A[i] * B[i];
        C[i] = A[i] / B[i];
    }
}

// Вспомогательная часть merge sort
void merge(float* arr, float* tmp, size_t l, size_t m, size_t r) {
    size_t i = l, j = m, k = l;
    while (i < m && j < r)
        tmp[k++] = (arr[i] < arr[j] ? arr[i++] : arr[j++]);
    while (i < m) tmp[k++] = arr[i++];
    while (j < r) tmp[k++] = arr[j++];
    for (i = l; i < r; ++i) arr[i] = tmp[i];
}

void cpu_merge_sort_rec(float* arr, float* tmp, size_t l, size_t r) {
    if (r - l <= 1) return;
    size_t m = l + (r - l) / 2;
    cpu_merge_sort_rec(arr, tmp, l, m);
    cpu_merge_sort_rec(arr, tmp, m, r);
    merge(arr, tmp, l, m, r);
}

void cpu_merge_sort(float* data, size_t N) {
    std::vector<float> tmp(N);
    cpu_merge_sort_rec(data, tmp.data(), 0, N);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <N>\n";
        return 1;
    }
    size_t N = std::stoul(argv[1]);
    if (N < 100000) {
        std::cerr << "Error: N must be >= 100000\n";
        return 1;
    }

    // инициализируем входные массивы
    std::vector<float> A(N), B(N), C(N);
    for (size_t i = 0; i < N; ++i) {
        A[i] = rand() / float(RAND_MAX);
        B[i] = rand() / float(RAND_MAX);
    }

    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    cpu_arith(A.data(), B.data(), C.data(), N);
    cpu_merge_sort(C.data(), N);
    auto t1 = clk::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    // единственная строка вывода — время в секундах
    std::cout << elapsed << "\n";
    return 0;
}
