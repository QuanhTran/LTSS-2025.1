#include <omp.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

using T = double;

void fill_random(std::vector<T>& a, unsigned seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(-1e6, 1e6);
    for (auto& x : a) x = dist(rng);
}

void selection_sort_seq(std::vector<T>& a) {
    const int n = static_cast<int>(a.size());
    for (int i = 0; i < n - 1; ++i) {
        int min_idx = i;
        T min_val = a[i];
        for (int j = i + 1; j < n; ++j) {
            if (a[j] < min_val) { min_val = a[j]; min_idx = j; }
        }
        if (min_idx != i) std::swap(a[i], a[min_idx]);
    }
}

// Parallel selection sort: parallelize the inner search for minimum using OpenMP.
// We keep one parallel region and at each i:
//  - threads split the j-range [i+1, n)
//  - each thread finds its local min (value+index)
//  - threads merge local minima into a shared (value+index) in a critical section
//  - a single thread performs the swap.
void selection_sort_omp(std::vector<T>& a) {
    const int n = static_cast<int>(a.size());
    T global_min_val{};
    int global_min_idx{};

#pragma omp parallel shared(global_min_val, global_min_idx, a)
    {
        for (int i = 0; i < n - 1; ++i) {
            // Initialize global min for this i (single thread)
#pragma omp single
            {
                global_min_val = a[i];
                global_min_idx = i;
            }

            // Each thread keeps its own local candidate
            T local_min_val = global_min_val;
            int local_min_idx = global_min_idx;

            // Split the search for min across threads
#pragma omp for nowait
            for (int j = i + 1; j < n; ++j) {
                const T v = a[j];
                if (v < local_min_val) { local_min_val = v; local_min_idx = j; }
            }

            // Merge local minima to global
#pragma omp critical
            {
                if (local_min_val < global_min_val) {
                    global_min_val = local_min_val;
                    global_min_idx = local_min_idx;
                }
            }

            // Ensure the global min is finalized before swapping
#pragma omp barrier
#pragma omp single
            {
                if (global_min_idx != i) std::swap(a[i], a[global_min_idx]);
            }
#pragma omp barrier
        }
    }
}

int main(int argc, char** argv) {
    // Default size is 100000 as requested. Allow override via argv[1].
    int N = (argc >= 1 + 1) ? std::max(1, std::atoi(argv[1])) : 100000;
    int threads = (argc >= 2 + 1) ? std::max(1, std::atoi(argv[2])) : 0; // 0 => OpenMP default

    if (threads > 0) omp_set_num_threads(threads);

    std::vector<T> base(N);
    fill_random(base);

    std::vector<T> a_seq = base;
    std::vector<T> a_omp = base;

    // Sequential
    auto t0 = std::chrono::steady_clock::now();
    selection_sort_seq(a_seq);
    auto t1 = std::chrono::steady_clock::now();
    double ms_seq = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // OpenMP
    auto t2 = std::chrono::steady_clock::now();
    selection_sort_omp(a_omp);
    auto t3 = std::chrono::steady_clock::now();
    double ms_omp = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Verify correctness
    bool ok = std::is_sorted(a_seq.begin(), a_seq.end())
           && std::equal(a_seq.begin(), a_seq.end(), a_omp.begin());

    std::cout << "Array size N = " << N << "\n";
    std::cout << "Threads (max) = " << omp_get_max_threads() << "\n";
    std::cout << "Sequential: " << ms_seq << " ms\n";
    std::cout << "OpenMP    : " << ms_omp << " ms\n";
    std::cout << "Speedup (seq/omp): " << (ms_seq / ms_omp) << "x\n";
    std::cout << "Correct   : " << (ok ? "YES" : "NO") << "\n";

    if (N >= 100000) {
        std::cout << "[Note] Selection sort is O(N^2). N=100000 may take a long time.\n";
        std::cout << "      You can test with smaller N via: ./selection_sort 20000\n";
    }
    return ok ? 0 : 1;
}
