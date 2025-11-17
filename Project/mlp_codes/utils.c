#include "utils.h"
#include <string.h>
#include <time.h>
#include <math.h>

// =====================================================
// Thread-safe Random Normal (Box–Muller)
// =====================================================
float randn(void)
{
#if defined(_OPENMP)
    unsigned int seed = 1234u * (unsigned int)omp_get_thread_num() ^ (unsigned int)time(NULL);

    float u = (rand_r(&seed) + 1.0f) / ((float)RAND_MAX + 2.0f);
    float v = (rand_r(&seed) + 1.0f) / ((float)RAND_MAX + 2.0f);
#else
    float u = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float v = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
#endif

    return sqrtf(-2.0f * logf(u)) * cosf(2.0f * M_PI * v);
}

float thread_randn(unsigned int *seed)
{
#ifdef _OPENMP
    unsigned int s = *seed;
    float u = (rand_r(&s) + 1.0f) / ((float)RAND_MAX + 2.0f);
    float v = (rand_r(&s) + 1.0f) / ((float)RAND_MAX + 2.0f);
    *seed = s;
#else
    float u = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float v = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
#endif

    return sqrtf(-2.0f * logf(u)) * cosf(2.0f * M_PI * v);
}


// =====================================================
// matmul: C = A[n×m] * B[m×p]
// =====================================================
void matmul(const float *restrict A,
            const float *restrict B,
            float *restrict C,
            int n, int m, int p)
{
    memset(C, 0, sizeof(float) * (size_t)(n * p));

    size_t work = (size_t)n * m * p;
    const size_t OMP_THRESHOLD = 60000;

    if (work < OMP_THRESHOLD)
    {
        for (int i = 0; i < n; i++)
        {
            int rA = i * m, rC = i * p;
            for (int k = 0; k < m; k++)
            {
                float a = A[rA + k];
                int rB = k * p;
                for (int j = 0; j < p; j++)
                    C[rC + j] += a * B[rB + j];
            }
        }
        return;
    }
    int num_threads = 1;
#if defined(_OPENMP)
    num_threads = omp_get_max_threads();
#endif

    int TILE = (n * m) / num_threads;
    if (TILE < 16)
        TILE = 16; // prevent too small tiles
    if (TILE > n)
        TILE = n; // avoid tile bigger than matrix

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i0 = 0; i0 < n; i0 += TILE)
        for (int k0 = 0; k0 < m; k0 += TILE)
            for (int j0 = 0; j0 < p; j0 += TILE)
            {
                int iMax = (i0 + TILE < n) ? (i0 + TILE) : n;
                int kMax = (k0 + TILE < m) ? (k0 + TILE) : m;
                int jMax = (j0 + TILE < p) ? (j0 + TILE) : p;

                for (int i = i0; i < iMax; i++)
                {
                    int rA = i * m, rC = i * p;
                    for (int k = k0; k < kMax; k++)
                    {
                        float a = A[rA + k];
                        int rB = k * p;
                        for (int j = j0; j < jMax; j++)
                            C[rC + j] += a * B[rB + j];
                    }
                }
            }
}

// =====================================================
// C = Aᵀ[M×N] * B[N×K]
// =====================================================
void matmul_Ta_b(const float *restrict A,
                 const float *restrict B,
                 float *restrict C,
                 int N, int M, int K)
{
    memset(C, 0, sizeof(float) * (size_t)(M * K));

    size_t work = (size_t)N * M * K;
    const size_t OMP_THRESHOLD = 60000;

    if (work < OMP_THRESHOLD)
    {
        for (int i = 0; i < M; i++)
        {
            int rC = i * K;
            for (int n = 0; n < N; n++)
            {
                float a = A[n * M + i];
                int rB = n * K;
                for (int j = 0; j < K; j++)
                    C[rC + j] += a * B[rB + j];
            }
        }
        return;
    }

    int num_threads = 1;
#if defined(_OPENMP)
    num_threads = omp_get_max_threads();
#endif

    // Dynamic tile size based on threads
    int TILE = (M * N) / num_threads;
    if (TILE < 16)
        TILE = 16;
    if (TILE > M)
        TILE = M;

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i0 = 0; i0 < M; i0 += TILE)
        for (int n0 = 0; n0 < N; n0 += TILE)
            for (int j0 = 0; j0 < K; j0 += TILE)
            {
                int iMax = (i0 + TILE < M) ? (i0 + TILE) : M;
                int nMax = (n0 + TILE < N) ? (n0 + TILE) : N;
                int jMax = (j0 + TILE < K) ? (j0 + TILE) : K;

                for (int i = i0; i < iMax; i++)
                {
                    int rC = i * K;
                    for (int n = n0; n < nMax; n++)
                    {
                        float a = A[n * M + i]; 
                        int rB = n * K;
                        for (int j = j0; j < jMax; j++)
                            C[rC + j] += a * B[rB + j];
                    }
                }
            }
}

// =====================================================
// out[j] = Σ_n mat[n][j]
// =====================================================
void reduce_sum_rows(const float *mat, float *out, int N, int D)
{
    for (int j = 0; j < D; j++)
        out[j] = 0.0f;

#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : out[ : D]) schedule(static)
#endif
    for (int n = 0; n < N; n++)
    {
        int r = n * D;
        for (int j = 0; j < D; j++)
            out[j] += mat[r + j];
    }
}

// =====================================================
// Add bias 
// =====================================================
void add_bias(float *Z, const float *b, int n, int p)
{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; i++)
    {
        int r = i * p;
        for (int j = 0; j < p; j++)
            Z[r + j] += b[j];
    }
}

// =====================================================
// Activation functions
// =====================================================
static float tanh_f(float x) { return tanhf(x); }
static float tanh_d(float a) { return 1.0f - a * a; }
Activation ACT_TANH = {tanh_f, tanh_d};

static float relu_f(float x) { return (x > 0) ? x : 0.0f; }
static float relu_d(float a) { return (a > 0) ? 1.0f : 0.0f; }
Activation ACT_RELU = {relu_f, relu_d};

static float leaky_relu_f(float x) { return (x > 0 ? x : 0.01f * x); }
static float leaky_relu_d(float a) { return (a > 0 ? 1.0f : 0.01f); }
Activation ACT_LEAKY_RELU = {leaky_relu_f, leaky_relu_d};

static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
static float sigmoid_d(float a) { return a * (1.0f - a); }
Activation ACT_SIGMOID = {sigmoid_f, sigmoid_d};

// =====================================================
// Stable softmax (subtract max for numerical safety)
// =====================================================
void softmax(const float *Z, float *P, int n, int p)
{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; i++)
    {
        int r = i * p;

        float maxv = Z[r];
        for (int j = 1; j < p; j++)
            if (Z[r + j] > maxv)
                maxv = Z[r + j];

        float s = 0.0f;
        for (int j = 0; j < p; j++)
        {
            float e = expf(Z[r + j] - maxv);
            P[r + j] = e;
            s += e;
        }

        float inv_s = 1.0f / s;
        for (int j = 0; j < p; j++)
            P[r + j] *= inv_s;
    }
}

// --------------------------------------------------------
// Learning rate schedules
// --------------------------------------------------------
float inverse_time_decay(float eta0, float k, int t)
{
    return eta0 / (1.0f + k * t);
}

float exponential_decay(float eta0, float k, int t)
{
    return eta0 * expf(-k * t);
}

// =====================================================
// File loading
// =====================================================
int count_lines(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
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

void load_X(const char *filename, float *X, int N, int D)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        perror("X");
        exit(1);
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            if (fscanf(fp, "%f", &X[i * D + j]) != 1)
            {
                fprintf(stderr, "Error reading X at %d,%d\n", i, j);
                exit(1);
            }
    fclose(fp);
}

void load_y(const char *filename, int *y, int N)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        perror("y");
        exit(1);
    }
    for (int i = 0; i < N; i++)
        if (fscanf(fp, "%d", &y[i]) != 1)
        {
            fprintf(stderr, "Error reading y at %d\n", i);
            exit(1);
        }
    fclose(fp);
}


