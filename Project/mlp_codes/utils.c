#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ----------------------
// Random helper
// ----------------------

// Normal distribution (Box-Muller)
float randn() {
    float u = ((float) rand() + 1.0f) / ((float) RAND_MAX + 1.0f);
    float v = ((float) rand() + 1.0f) / ((float) RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u)) * cosf(2.0f * 3.14159265358979323846f * v);
}

// ----------------------
// Basic tensor ops
// ----------------------

// C[n×p] = A[n×m] * B[m×p]
void transpose(const float *B, float *B_T, int m, int p) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++)
            B_T[j*m + i] = B[i*p + j];
}

void matmul(const float *A, const float *B_T, float *C, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        int rowA = i * m;
        int rowC = i * p;

        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            int rowB = j * m;

            for (int k = 0; k < m; k++)
                sum += A[rowA + k] * B_T[rowB + k];

            C[rowC + j] = sum;
        }
    }
}


// Add bias b[p] to each row of Z[n×p]
void add_bias(float *Z, float *b, int n, int p) {
    for (int i = 0; i < n; i++) {
        int row = i * p;
        for (int j = 0; j < p; j++)
            Z[row + j] += b[j];
    }
}

// Softmax row-by-row
void softmax(float *Z, float *P, int n, int p) {
    for (int i = 0; i < n; i++) {
        int row = i * p;

        float sum_exp = 0.0f;
        for (int j = 0; j < p; j++) {
            P[row + j] = expf(Z[row + j]); 
            sum_exp += P[row+j];
        }
        float inv = 1.0f / sum_exp;

        for (int j = 0; j < p; j++)
            P[row + j] *= inv;
    }
}

// ----------------------
// Data loading
// ----------------------

int count_lines(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("File open error");
        exit(EXIT_FAILURE);
    }

    int count = 0;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), fp))
        count++;

    fclose(fp);
    return count;
}

void load_X(const char *filename, float *X, int num_examples, int input_dim) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening X");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_examples; i++)
        for (int j = 0; j < input_dim; j++)
            if (fscanf(fp, "%f", &X[i*input_dim + j]) != 1) {
                fprintf(stderr, "Error reading X row %d col %d\n", i, j);
                fclose(fp);
                exit(EXIT_FAILURE);
            }

    fclose(fp);
}

void load_y(const char *filename, int *y, int num_examples) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening y");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_examples; i++)
        if (fscanf(fp, "%d", &y[i]) != 1) {
            fprintf(stderr, "Error reading y at line %d\n", i);
            fclose(fp);
            exit(EXIT_FAILURE);
        }

    fclose(fp);
}
