#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// -------------------------
// Activation structure
// -------------------------
typedef struct {
    float (*func)(float);   // Activation function
    float (*deriv)(float);  // Derivative (computed from activation output)
} Activation;

// Built-in activations
extern Activation ACT_TANH;
extern Activation ACT_RELU;
extern Activation ACT_SIGMOID;

// -------------------------
// Random
// -------------------------
float randn();

// -------------------------
// Matrix operations
// -------------------------
void matmul(float *A, float *B, float *C, int n, int m, int p);
void matmul_Ta_b(const float *A, const float *B, float *C, int N, int M, int K);
void reduce_sum_rows(const float *mat, float *out, int N, int D);
void add_bias(float *Z, float *b, int n, int p);

// -------------------------
// Activation helpers
// -------------------------
void tanh_activation(float *a1, int num_examples, int nn_hdim, float *dtanh);
void softmax(float *Z, float *P, int n, int p);

// -------------------------
// File loading utilities
// -------------------------
int count_lines(const char *filename);
void load_X(const char *filename, float *X, int num_examples, int input_dim);
void load_y(const char *filename, int *y, int num_examples);

#endif
