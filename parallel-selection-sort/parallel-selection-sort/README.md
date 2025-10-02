# Parallel Selection Sort (OpenMP) — VS Code Project

This project sorts an array of floating-point numbers using **Selection Sort**:
1) Sequential implementation
2) OpenMP-parallel implementation (inner min-search with a thread-local reduction)

It measures execution time and prints the speedup and a correctness check.

> **Warning:** Selection sort is O(N^2). With N=100,000, it can take a long time.
> Start with N=20,000 to verify, then try larger sizes if desired.

## Build & Run (VS Code)
- **Ctrl+Shift+B** → build (task **build-openmp**).
- Command Palette → **Run Task** → **build-and-run quick** (N=20,000).
- To run the full size: run task **run (N=100000)** after building.

## Command-line (Linux/WSL)
```
g++ -O3 -march=native -fopenmp selection_sort.cpp -o build/selection_sort
./build/selection_sort 20000 0     # quick test
./build/selection_sort 100000 0    # full size (slow)
```

## Command-line (Windows MSYS2 MINGW64)
```
C:/msys64/mingw64/bin/g++.exe -O3 -fopenmp selection_sort.cpp -o build/selection_sort.exe
./build/selection_sort.exe 20000 0
```

Arguments: `N threads`
- `N` (default 100000): array size.
- `threads` (default 0): if >0, sets `OMP_NUM_THREADS` for the OpenMP run.

The OpenMP version keeps a single parallel region and, for each outer index `i`,
parallelizes the search of the minimum in `a[i+1..n)` using per-thread local minima
and a critical section to combine them, then performs the swap in a `single` region.
