#include <omp.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using T = double;
constexpr int ROWS = 100;
constexpr int COLS = 2000;
constexpr size_t N = static_cast<size_t>(ROWS) * COLS;

void fill_data(std::vector<T>& A, std::vector<T>& B) {
    for (size_t i = 0; i < N; ++i) {
        A[i] = static_cast<T>(i) * 0.001;
        B[i] = static_cast<T>(i) * 0.002;
    }
}

void add_seq(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C) {
    for (size_t i = 0; i < N; ++i) C[i] = A[i] + B[i];
}

void add_omp(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i) C[i] = A[i] + B[i];
}

int main(int argc, char** argv) {
    int repeats = (argc >= 2) ? std::max(1, std::atoi(argv[1])) : 1000; // number of repetitions
    int threads = (argc >= 3) ? std::max(1, std::atoi(argv[2])) : 0;    // 0 -> use OMP defaults

    if (threads > 0) omp_set_num_threads(threads);

    std::vector<T> A(N), B(N), C_seq(N), C_omp(N);
    fill_data(A, B);

    // Sequential
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) add_seq(A, B, C_seq);
    auto t1 = std::chrono::steady_clock::now();
    double ms_seq = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // OpenMP
    auto t2 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) add_omp(A, B, C_omp);
    auto t3 = std::chrono::steady_clock::now();
    double ms_omp = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Verify
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double d = std::fabs(C_seq[i] - C_omp[i]);
        if (d > max_abs_diff) max_abs_diff = d;
    }

    int used_threads = omp_get_max_threads();
    std::cout << "Matrix size: " << ROWS << "x" << COLS << " (N=" << N << ")\n";
    std::cout << "Repeats: " << repeats << "\n";
    std::cout << "Threads (max): " << used_threads << "\n";
    std::cout << "Sequential time: " << ms_seq << " ms\n";
    std::cout << "OpenMP time    : " << ms_omp << " ms\n";
    std::cout << "Speedup (seq/omp): " << (ms_seq / ms_omp) << "x\n";
    std::cout << "Max |diff|: " << max_abs_diff << "\n";
    return 0;
}
