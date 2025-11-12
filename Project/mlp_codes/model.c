#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// --------------------------------------------------------
void init_weights(MLP *m)
{
    for (int i = 0; i < m->n_in * m->n_hidden; i++)
        m->W1[i] = randn() / sqrtf(m->n_in);

    for (int i = 0; i < m->n_hidden * m->n_out; i++)
        m->W2[i] = randn() / sqrtf(m->n_hidden);

    memset(m->b1, 0, m->n_hidden * sizeof(float));
    memset(m->b2, 0, m->n_out * sizeof(float));
}

// --------------------------------------------------------
void forward_pass(
    MLP *m, float *X, int N,
    float *z1, float *a1, float *z2, float *probs)
{
    // z1 = X*W1 + b1
    matmul(X, m->W1, z1, N, m->n_in, m->n_hidden);
    add_bias(z1, m->b1, N, m->n_hidden);

    // a1 = activation(z1)
    for (int i = 0; i < N * m->n_hidden; i++)
        a1[i] = m->act.func(z1[i]);

    // z2 = a1*W2 + b2
    matmul(a1, m->W2, z2, N, m->n_hidden, m->n_out);
    add_bias(z2, m->b2, N, m->n_out);

    softmax(z2, probs, N, m->n_out);
}

// --------------------------------------------------------
float compute_loss(MLP *m, float *X, int *y, int N, float reg_lambda)
{
    float *z1 = calloc(N * m->n_hidden, sizeof(float));
    float *a1 = calloc(N * m->n_hidden, sizeof(float));
    float *z2 = calloc(N * m->n_out, sizeof(float));
    float *probs = calloc(N * m->n_out, sizeof(float));

    forward_pass(m, X, N, z1, a1, z2, probs);

    float loss = 0.0f;
    for (int i = 0; i < N; i++)
        loss += -logf(probs[i * m->n_out + y[i]]);

    // L2 regularization
    float reg = 0.0f;
    for (int i = 0; i < m->n_in * m->n_hidden; i++)
        reg += m->W1[i] * m->W1[i];
    for (int i = 0; i < m->n_hidden * m->n_out; i++)
        reg += m->W2[i] * m->W2[i];

    loss = (loss + 0.5f * reg_lambda * reg) / N;

    free(z1);
    free(a1);
    free(z2);
    free(probs);
    return loss;
}

void update_gradient(MLP *m, float *dW1, float *dW2, float *db1, float *db2, float lr)
{
    int L1 = m->n_in * m->n_hidden;
    int L2 = m->n_hidden * m->n_out;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < L1; i++)
        m->W1[i] -= lr * dW1[i];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < L2; i++)
        m->W2[i] -= lr * dW2[i];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < m->n_hidden; i++)
        m->b1[i] -= lr * db1[i];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < m->n_out; i++)
        m->b2[i] -= lr * db2[i];
}

// --------------------------------------------------------
// Training: backprop + gradient descent (MPI-aware)
// --------------------------------------------------------
void train(
    MLP *m, float *X, int *y, int N,
    float lr, float reg_lambda,
    int batch_size,
    int num_passes, int print_loss)
{
    int world_rank = 0, world_size = 1;
    int mpi_initialized = 0;
    if (MPI_Initialized(&mpi_initialized) == MPI_SUCCESS && mpi_initialized)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    if (batch_size <= 0)
        batch_size = N;
    if (batch_size > N)
        batch_size = N;

    int num_batches = (N + batch_size - 1) / batch_size;
    int L1 = m->n_in * m->n_hidden;
    int L2 = m->n_hidden * m->n_out;

    // ---- Allocate all working buffers ONCE ----
    float *z1 = calloc(batch_size * m->n_hidden, sizeof(float));
    float *a1 = calloc(batch_size * m->n_hidden, sizeof(float));
    float *z2 = calloc(batch_size * m->n_out, sizeof(float));
    float *probs = calloc(batch_size * m->n_out, sizeof(float));

    float *delta3 = malloc(batch_size * m->n_out * sizeof(float));
    float *delta2 = malloc(batch_size * m->n_hidden * sizeof(float));

    float *dW1 = calloc(L1, sizeof(float));
    float *dW2 = calloc(L2, sizeof(float));
    float *db1 = calloc(m->n_hidden, sizeof(float));
    float *db2 = calloc(m->n_out, sizeof(float));

    float *Xb = malloc(batch_size * m->n_in * sizeof(float));
    int *yb = malloc(batch_size * sizeof(int));

    for (int epoch = 0; epoch < num_passes; ++epoch)
    {
        for (int b = 0; b < num_batches; ++b)
        {
            int start = b * batch_size;
            int bs = ((start + batch_size) <= N) ? batch_size : (N - start);

            // --- Copy batch data ---
            for (int i = 0; i < bs; i++)
            {
                int src = start + i;
                yb[i] = y[src];
                memcpy(&Xb[i * m->n_in], &X[src * m->n_in], m->n_in * sizeof(float));
            }

            // --- Forward pass ---
            forward_pass(m, Xb, bs, z1, a1, z2, probs);

            // --- Backward pass ---
            memcpy(delta3, probs, bs * m->n_out * sizeof(float));
            for (int i = 0; i < bs; i++)
                delta3[i * m->n_out + yb[i]] -= 1.0f;

            memset(dW1, 0, L1 * sizeof(float));
            memset(dW2, 0, L2 * sizeof(float));
            memset(db1, 0, m->n_hidden * sizeof(float));
            memset(db2, 0, m->n_out * sizeof(float));

            // Gradients
            matmul_Ta_b(a1, delta3, dW2, bs, m->n_hidden, m->n_out);
            reduce_sum_rows(delta3, db2, bs, m->n_out);

            matmul(delta3, m->W2, delta2, bs, m->n_out, m->n_hidden);
            for (int i = 0; i < bs * m->n_hidden; i++)
                delta2[i] *= m->act.deriv(a1[i]);

            matmul_Ta_b(Xb, delta2, dW1, bs, m->n_in, m->n_hidden);
            reduce_sum_rows(delta2, db1, bs, m->n_hidden);

            // Normalize + regularize
            float inv_bs = 1.0f / (float)bs;
            for (int i = 0; i < L1; i++)
                dW1[i] = dW1[i] * inv_bs + reg_lambda * m->W1[i];
            for (int i = 0; i < L2; i++)
                dW2[i] = dW2[i] * inv_bs + reg_lambda * m->W2[i];
            for (int i = 0; i < m->n_hidden; i++)
                db1[i] *= inv_bs;
            for (int i = 0; i < m->n_out; i++)
                db2[i] *= inv_bs;

            // --- MPI gradient reduction (if MPI is active) ---
            if (mpi_initialized && world_size > 1)
            {
                MPI_Allreduce(MPI_IN_PLACE, dW1, L1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, dW2, L2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, db1, m->n_hidden, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, db2, m->n_out, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

                float inv = 1.0f / (float)world_size;
                for (int i = 0; i < L1; i++) dW1[i] *= inv;
                for (int i = 0; i < L2; i++) dW2[i] *= inv;
                for (int i = 0; i < m->n_hidden; i++) db1[i] *= inv;
                for (int i = 0; i < m->n_out; i++) db2[i] *= inv;
            }

            // Apply update
            update_gradient(m, dW1, dW2, db1, db2, lr);
        }

        if (print_loss && (epoch % 100 == 0 || epoch == num_passes - 1))
        {
            // Only rank 0 should print final loss in an MPI run
            int do_print = 1;
            if (mpi_initialized && world_size > 1) {
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                do_print = (rank == 0);
            }
            if (do_print) {
                float L = compute_loss(m, X, y, N, reg_lambda);
                printf("Epoch %d / %d  Loss: %.6f\n", epoch, num_passes, L);
            }
        }
    }

    // ---- Free once after training ----
    free(z1);
    free(a1);
    free(z2);
    free(probs);
    free(delta3);
    free(delta2);
    free(dW1);
    free(dW2);
    free(db1);
    free(db2);
    free(Xb);
    free(yb);
}

// --------------------------------------------------------
// Create and free model
// --------------------------------------------------------
MLP *create_model(int n_in, int n_hidden, int n_out, Activation act)
{
    MLP *m = malloc(sizeof(MLP));
    if (!m) return NULL;
    m->n_in = n_in;
    m->n_hidden = n_hidden;
    m->n_out = n_out;
    m->act = act;

    m->W1 = malloc((size_t)n_in * n_hidden * sizeof(float));
    m->b1 = calloc(n_hidden, sizeof(float));
    m->W2 = malloc((size_t)n_hidden * n_out * sizeof(float));
    m->b2 = calloc(n_out, sizeof(float));

    if (!m->W1 || !m->b1 || !m->W2 || !m->b2) {
        free(m->W1); free(m->b1); free(m->W2); free(m->b2); free(m);
        return NULL;
    }

    init_weights(m);
    return m;
}

void free_model(MLP *m)
{
    if (!m)
        return;
    free(m->W1);
    free(m->b1);
    free(m->W2);
    free(m->b2);
    free(m);
}
