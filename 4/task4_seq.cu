#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath> // Для fabs

using namespace std;
using namespace chrono;

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <rows> <cols> <operation>\n";
        cerr << "Operations: 0-add, 1-sub, 2-mul, 3-div\n";
        return 1;
    }

    const size_t rows = stoul(argv[1]);
    const size_t cols = stoul(argv[2]);
    const int op = stoi(argv[3]);
    const size_t N = rows * cols;
    
    if (N < 100000) {
        cerr << "Error: Total elements must be >= 100000\n";
        return 1;
    }

    // Инициализация массивов
    vector<float> A(N), B(N), C(N);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const size_t idx = i * cols + j;
            A[idx] = (i * cols + j + 1) * 0.1f;
            B[idx] = (N - i * cols - j) * 0.1f;
            if (op == 3 && fabs(B[idx]) < 1e-6f) B[idx] = 1.0f; // Защита от деления на 0
        }
    }

    auto t0 = high_resolution_clock::now();
    
    // Выполнение операций
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const size_t idx = i * cols + j;
            switch(op) {
                case 0: C[idx] = A[idx] + B[idx]; break;
                case 1: C[idx] = A[idx] - B[idx]; break;
                case 2: C[idx] = A[idx] * B[idx]; break;
                case 3: C[idx] = A[idx] / B[idx]; break;
            }
        }
    }
    
    auto t1 = high_resolution_clock::now();
    double elapsed = duration_cast<duration<double>>(t1 - t0).count();
    
    // Проверка результата (первый и последний элемент)
    cout << elapsed << "\n";
    return 0;
}