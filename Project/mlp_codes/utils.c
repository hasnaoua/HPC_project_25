#include "utils.h"

// -------------------------
// Random Normal
// -------------------------
float randn() {
    float u = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float v = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u)) * cosf(2.0f * M_PI * v);
}

// -------------------------
// Matrix multiplication
// -------------------------
void matmul(float *A, float *B, float *C, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        int rA = i * m;
        int rC = i * p;
        for (int j = 0; j < p; j++) {
            float s = 0;
            for (int k = 0; k < m; k++)
                s += A[rA + k] * B[k * p + j];
            C[rC + j] = s;
        }
    }
}

// Multiply transpose(A) * B
void matmul_Ta_b(const float *A, const float *B, float *C, int N, int M, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float s = 0;
            for (int n = 0; n < N; n++)
                s += A[n * M + i] * B[n * K + j];
            C[i * K + j] = s;
        }
    }
}

// Sum rows of a matrix
void reduce_sum_rows(const float *mat, float *out, int N, int D) {
    for (int j = 0; j < D; j++)
        out[j] = 0;

    for (int n = 0; n < N; n++) {
        int r = n * D;
        for (int j = 0; j < D; j++)
            out[j] += mat[r + j];
    }
}

// Add bias vector to each row of matrix
void add_bias(float *Z, float *b, int n, int p) {
    for (int i = 0; i < n; i++) {
        int r = i * p;
        for (int j = 0; j < p; j++)
            Z[r + j] += b[j];
    }
}

// -------------------------
// Activations
// -------------------------
// Tanh
static float tanh_f(float x) { return tanhf(x); }
static float tanh_d(float a) { return 1.0f - a * a; }
Activation ACT_TANH = { tanh_f, tanh_d };

// ReLU
static float relu_f(float x) { return x > 0 ? x : 0; }
static float relu_d(float a) { return a > 0 ? 1.0f : 0.0f; }
Activation ACT_RELU = { relu_f, relu_d };

// Sigmoid
static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
static float sigmoid_d(float a) { return a * (1.0f - a); }
Activation ACT_SIGMOID = { sigmoid_f, sigmoid_d };

// Compute derivative of tanh for a matrix
void tanh_activation(float *a1, int N, int H, float *dt) {
    int sz = N * H;
    for (int i = 0; i < sz; i++)
        dt[i] = 1.0f - a1[i] * a1[i];
}

// Softmax
void softmax(float *Z, float *P, int n, int p) {
    for (int i = 0; i < n; i++) {
        int r = i * p;
        float s = 0;

        // Exponentiate and sum
        for (int j = 0; j < p; j++) {
            P[r + j] = expf(Z[r + j]);
            s += P[r + j];
        }

        // Normalize
        float inv = 1.0f / s;
        for (int j = 0; j < p; j++)
            P[r + j] *= inv;
    }
}

// -------------------------
// File utilities
// -------------------------
int count_lines(const char *fn) {
    FILE *fp = fopen(fn, "r");
    if (!fp) {
        perror("open");
        exit(1);
    }

    int c = 0;
    char buf[4096];
    while (fgets(buf, sizeof(buf), fp))
        c++;
    fclose(fp);
    return c;
}

void load_X(const char *f, float *X, int N, int D) {
    FILE *fp = fopen(f, "r");
    if (!fp) {
        perror("X");
        exit(1);
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            if (fscanf(fp, "%f", &X[i * D + j]) != 1) {
                fprintf(stderr, "Error reading X at %d,%d\n", i, j);
                exit(1);
            }
    fclose(fp);
}

void load_y(const char *f, int *y, int N) {
    FILE *fp = fopen(f, "r");
    if (!fp) {
        perror("y");
        exit(1);
    }

    for (int i = 0; i < N; i++)
        if (fscanf(fp, "%d", &y[i]) != 1) {
            fprintf(stderr, "Error reading y at %d\n", i);
            exit(1);
        }

    fclose(fp);
}
