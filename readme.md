# Quadratic Equation Solver - High performance computing

A high-performance solver for quadratic equations (`ax² + bx + c = 0`) with multiple implementations for comparison.

## Project Description

This project solves millions of quadratic equations using different optimization techniques:
- Basic CPU version
- CPU with OpenMP multithreading
- SSE2 vectorization
- SSE2 + OpenMP combination
- CUDA GPU acceleration

All implementations are verified to produce identical mathematical results.

## Performance Results

| Implementation  | Cycles   | Speed Gain |
|-----------------|----------|------------|
| Basic CPU       | 396M     | 1x         |
| CPU + OpenMP    | 58.8M    | 6.73x      |
| SSE2            | 50.6M    | 7.83x      |
| SSE2 + OpenMP   | 31.3M    | 12.63x     |
| CUDA        | 7.9M | 49.84x |

## Sample output

```cpp
Cuda    Polynome n 19999997:
Cuda            8.8861 . x^2 - 1.1454 . x + 4.8112 = 0.0
Cuda                    Solution 1 : 0.0644 + i . 0.7330
Cuda                    Solution 2 : 0.0644 - i . 0.7330

Duree cuda           : 7952104 cycles - Gain = 49.84
```