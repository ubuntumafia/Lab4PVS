#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

void merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    while (i <= mid && j <= right)
        temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    for (int t = 0; t < k; ++t)
        arr[left + t] = temp[t];
}

void mergeSort(std::vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int mid = (left + right) / 2;
    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Использование: " << argv[0] << " <array_size>\n";
        return 1;
    }

    int n = atoi(argv[1]);
    std::vector<int> arr(n);
    for (int i = 0; i < n; ++i)
        arr[i] = rand();

    auto start = std::chrono::high_resolution_clock::now();
    mergeSort(arr, 0, n - 1);
    auto end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Последовательная сортировка завершена\n";
    std::cout << "Время выполнения: " << duration << " секунд\n";
    return 0;
}
