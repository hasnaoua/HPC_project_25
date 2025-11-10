#include "utils.h"
#include <string.h> 

// -------------------------
// Random Normal (Box-Muller)
// -------------------------
float randn() {
    float u = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float v = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u)) * cosf(2.0f * M_PI * v);
}

// =====================================================
// Matrix multiplication (A[n×m] * B[m×p] = C[n×p])
// =====================================================
void matmul(const float *restrict A,
            const float *restrict B,
            float *restrict C,
            int n, int m, int p)
{
    /* Hoist frequently-used local variables outside loops.
       Initialize each row of C then compute row by row.
       We load a = A[rA + k] once per k and reuse it across the inner j loop.
    */

    int i, k, j;
    int rA = 0, rB = 0, rC = 0;
    float a;

    for (i = 0; i < n; ++i) {
        rA = i * m;
        rC = i * p;

        /* initialize row of C to 0 */
        memset(&C[rC], 0, sizeof(float) * (size_t)p);

        for (k = 0; k < m; ++k) {
            a = A[rA + k];
            rB = k * p;

            /* inner loop: add a * B[rB + j] into C[rC + j] */
            for (j = 0; j < p; ++j)
                C[rC + j] += a * B[rB + j];
        }
    }
}

// =====================================================
// Multiply transpose(A) * B  (same version you gave)
// A is N×M, B is N×K, result C is M×K
// =====================================================
void matmul_Ta_b(const float *restrict A,
                 const float *restrict B,
                 float *restrict C,
                 int N, int M, int K)
{
    // Initialize C
    memset(C, 0, sizeof(float) * (size_t)M * (size_t)K);

    int rowA=0, rowB=0, rowC=0;
    float a;

    for (int i = 0; i < M; i++) {
        rowC = i * K;

        for (int n = 0; n < N; n++) {
            rowA = n * M;
            rowB = n * K;

            a = A[rowA + i];

            for (int j = 0; j < K; j++)
                C[rowC + j] += a * B[rowB + j];
        }
    }
}

// =====================================================
// Reduce sum over rows: out[d] = sum_n mat[n,d]
// =====================================================
void reduce_sum_rows(const float *mat, float *out, int N, int D)
{
    for (int j = 0; j < D; j++)
        out[j] = 0.0f;

    for (int n = 0; n < N; n++) {
        int r = n * D;
        for (int j = 0; j < D; j++)
            out[j] += mat[r + j];
    }
}

// =====================================================
// Add bias vector to each row
// =====================================================
void add_bias(float *Z, float *b, int n, int p)
{
    for (int i = 0; i < n; i++) {
        int r = i * p;
        for (int j = 0; j < p; j++)
            Z[r + j] += b[j];
    }
}

// =====================================================
// Activations
// =====================================================

// Tanh
static float tanh_f(float x) { return tanhf(x); }
static float tanh_d(float a) { return 1.0f - a * a; }
Activation ACT_TANH = {tanh_f, tanh_d};

// ReLU
static float relu_f(float x) { return x > 0 ? x : 0.0f; }
static float relu_d(float a) { return a > 0 ? 1.0f : 0.0f; }
Activation ACT_RELU = {relu_f, relu_d};

// Sigmoid
static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
static float sigmoid_d(float a) { return a * (1.0f - a); }
Activation ACT_SIGMOID = {sigmoid_f, sigmoid_d};

// =====================================================
// Tanh derivative over matrix (uses tanh_d now)
// =====================================================
void tanh_activation(float *a1, int N, int H, float *dt)
{
    int sz = N * H;
    for (int i = 0; i < sz; i++)
        dt[i] = tanh_d(a1[i]);  
}

// =====================================================
// Softmax
// =====================================================
void softmax(float *Z, float *P, int n, int p)
{
    for (int i = 0; i < n; i++) {
        int r = i * p;
        float s = 0.0f;

        for (int j = 0; j < p; j++) {
            P[r + j] = expf(Z[r + j]);
            s += P[r + j];
        }

        float inv = 1.0f / s;
        for (int j = 0; j < p; j++)
            P[r + j] *= inv;
    }
}

// =====================================================
// File utilities
// =====================================================
int count_lines(const char *fn)
{
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

void load_X(const char *f, float *X, int N, int D)
{
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

void load_y(const char *f, int *y, int N)
{
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
