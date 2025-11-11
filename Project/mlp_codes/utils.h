#ifndef UTILS_H
#define UTILS_H

// Define PI if not provided
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Random normal generator
float randn();

// Basic tensor ops
void matmul(const float *A, const float *B_T, float *C, int n, int m, int p);
void transpose(const float *B, float *B_T, int m, int p);
void add_bias(float *Z, float *b, int n, int p);
void softmax(float *Z, float *P, int n, int p);

// Data loading
int count_lines(const char *filename);
void load_X(const char *filename, float *X, int num_examples, int input_dim);
void load_y(const char *filename, int *y, int num_examples);

#endif
