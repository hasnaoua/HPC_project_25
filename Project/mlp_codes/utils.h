#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// =====================================================
// Activation structure
// =====================================================
typedef struct
{
    float (*func)(float);   // activation: f(x)
    float (*deriv)(float);  // derivative using output a = f(x)
} Activation;

// Built-in activations (defined in utils.c)
extern Activation ACT_TANH;
extern Activation ACT_RELU;
extern Activation ACT_SIGMOID;
extern Activation ACT_LEAKY_RELU;

// =====================================================
// Random utilities
// =====================================================
// Gaussian random number using Box–Muller
float randn(void);

// =====================================================
// Matrix operations
// =====================================================
// C = A (n×m) @ B (m×p)
void matmul(const float *restrict A,
            const float *restrict B,
            float       *restrict C,
            int n, int m, int p);

// C = A^T (M×N) @ B (M×K)
// A stored as (N×M) row-major but interpreted transposed
void matmul_Ta_b(const float *restrict A,
                 const float *restrict B,
                 float       *restrict C,
                 int N, int M, int K);

// out[d] = sum over rows of mat[:, d]
void reduce_sum_rows(const float *mat,
                     float *out,
                     int N, int D);

// Add bias row-wise:
// Z[i, j] += b[j]
void add_bias(float *Z,
              const float *b,
              int N, int D);

// =====================================================
// Softmax
// =====================================================
void softmax(const float *Z,
             float *P,
             int N, int D);

// =====================================================
// Learning rate schedules
// =====================================================
typedef float (*LRSchedule)(float eta0, float k, int t);

float inverse_time_decay(float eta0, float k, int t);
float exponential_decay  (float eta0, float k, int t);

// =====================================================
// File utilities
// =====================================================
int  count_lines(const char *filename);
void load_X(const char *filename, float *X, int N, int D);
void load_y(const char *filename, int *y, int N);

#endif // UTILS_H
