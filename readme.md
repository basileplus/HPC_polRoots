# Quadratic Equation Solver - High performance computing

A high-performance solver for quadratic equations ($a^2 + bx + c = 0$) with multiple implementations for comparison. The objective is to compare different way to increase computing performance.

## Project Description

This project solves millions of quadratic equations using different optimization techniques:
- Basic CPU version
- CPU with OpenMP multithreading
- SSE2 vectorization
- SSE2 + OpenMP combination
- CUDA GPU acceleration

## Performance Results
The compute has been done to solve 20M different polynoms randomly initialized with coeffs in $[-10,10]$
| Implementation  | Cycles   | Speed Gain |
|-----------------|----------|------------|
| Basic CPU       | 396M     | 1x         |
| CPU + OpenMP    | 58.8M    | 6.73x      |
| SSE2            | 50.6M    | 7.83x      |
| SSE2 + OpenMP   | 31.3M    | 12.63x     |
| CUDA        | 7.9M | 49.84x |

## Sample output

```
    Polynome n 19999997:
            8.8861 . x^2 - 1.1454 . x + 4.8112 = 0.0
                    Solution 1 : 0.0644 + i . 0.7330
                    Solution 2 : 0.0644 - i . 0.7330
```
