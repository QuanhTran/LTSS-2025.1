# Parallel Matrix Addition (OpenMP) — VS Code Project

This project adds two matrices of size 100x2000 using:
1) Sequential C++ loop
2) OpenMP parallel loop

It measures execution time and prints speedup.

## Build & Run (VS Code)
- Press **Ctrl+Shift+B** to run the default task **build-openmp**.
- Then run task **run**, or run the composite task **build-and-run** from the Command Palette.
- To debug, press **F5** (launch config: *Debug matrix_add*).

### Command-line (Linux/WSL)
```
g++ -O3 -march=native -fopenmp matrix_add.cpp -o build/matrix_add
./build/matrix_add 1000 0
```

### Command-line (Windows MSYS2 MINGW64)
```
C:/msys64/mingw64/bin/g++.exe -O3 -fopenmp matrix_add.cpp -o build/matrix_add.exe
./build/matrix_add.exe 1000 0
```

Arguments: `repeats threads`
- `repeats` (default 1000): how many times to repeat the add to get stable timings.
- `threads` (default 0): if >0, sets OMP threads explicitly.

You can also control threads with env var: `OMP_NUM_THREADS`.
