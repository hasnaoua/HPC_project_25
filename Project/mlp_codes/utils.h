#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif


#define TILE 16

#ifdef _OPENMP
#include <omp.h>
#endif

// =====================================================
// Activation structure
// =====================================================
typedef struct
{
    float (*func)(float);   // Activation function: f(x)
    float (*deriv)(float);  // Derivative using activation output a
} Activation;

// Built-in activations
extern Activation ACT_TANH;
extern Activation ACT_RELU;
extern Activation ACT_SIGMOID;

// =====================================================
// Random utilities
// =====================================================
float randn(void); // Normal distribution (Box–Muller)

// =====================================================
// Matrix operations
// =====================================================
void matmul(const float *restrict A,
            const float *restrict B,
            float *restrict C,
            int n, int m, int p);

void matmul_Ta_b(const float *restrict A,
                 const float *restrict B,
                 float *restrict C,
                 int N, int M, int K);

void reduce_sum_rows(const float *mat,
                     float *out,
                     int N, int D);

void add_bias(float *Z, const float *b,
              int n, int p);

// =====================================================
// Activation helpers
// =====================================================
void tanh_activation(const float *a1,
                     int N, int H,
                     float *dt);

void softmax(const float *Z,
             float *P,
             int n, int p);

// =====================================================
// File utilities
// =====================================================
int count_lines(const char *filename);
void load_X(const char *filename, float *X, int N, int D);
void load_y(const char *filename, int *y, int N);

#endif // UTILS_H
