// model.c (updated)
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------- Local thread-safe gaussian generator (Box-Muller) ----------
static inline float thread_rand_uniform(unsigned int *state)
{
    // return a float in (0,1], avoid 0
    int r = rand_r(state);
    return (r + 1.0f) / ((float)RAND_MAX + 2.0f);
}

static inline float thread_randn(unsigned int *state)
{
    // Box-Muller transform, produce a single N(0,1)
    // We use two uniforms, convert, and return cos branch.
    // Note: not optimized to reuse second sample; simple and safe.
    float u1 = thread_rand_uniform(state);
    float u2 = thread_rand_uniform(state);
    float radius = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * (float)M_PI * u2;
    return radius * cosf(theta);
}

// --------------------------------------------------------
void init_weights(MLP *m)
{
    if (!m)
        return;

#ifdef _OPENMP
    // Seed base derived from address + time to reduce identical seeds across processes
    unsigned int time_seed = (unsigned int)time(NULL);
    // Use parallel loop with each thread having its own rand state
#pragma omp parallel
    {
        unsigned int seed = time_seed + (unsigned int)(uintptr_t)m + (unsigned int)omp_get_thread_num() * 9973u;
#pragma omp for
        for (int i = 0; i < m->n_in * m->n_hidden; i++)
            m->W1[i] = thread_randn(&seed) / sqrtf((float)m->n_in);

#pragma omp for
        for (int i = 0; i < m->n_hidden * m->n_out; i++)
            m->W2[i] = thread_randn(&seed) / sqrtf((float)m->n_hidden);
    }
#else
    unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)(uintptr_t)m;
    for (int i = 0; i < m->n_in * m->n_hidden; i++)
        m->W1[i] = thread_randn(&seed) / sqrtf((float)m->n_in);
    for (int i = 0; i < m->n_hidden * m->n_out; i++)
        m->W2[i] = thread_randn(&seed) / sqrtf((float)m->n_hidden);
#endif

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
    if (N <= 0)
        return 0.0f;

    float *z1 = calloc((size_t)N * m->n_hidden, sizeof(float));
    float *a1 = calloc((size_t)N * m->n_hidden, sizeof(float));
    float *z2 = calloc((size_t)N * m->n_out, sizeof(float));
    float *probs = calloc((size_t)N * m->n_out, sizeof(float));
    if (!z1 || !a1 || !z2 || !probs)
    {
        free(z1);
        free(a1);
        free(z2);
        free(probs);
        return INFINITY;
    }

    forward_pass(m, X, N, z1, a1, z2, probs);

    float loss = 0.0f;
    for (int i = 0; i < N; i++)
    {
        float p = probs[i * m->n_out + y[i]];
        // numerical safety
        loss += -logf(fmaxf(p, 1e-12f));
    }

    // L2 regularization
    float reg = 0.0f;
    for (int i = 0; i < m->n_in * m->n_hidden; i++)
        reg += m->W1[i] * m->W1[i];
    for (int i = 0; i < m->n_hidden * m->n_out; i++)
        reg += m->W2[i] * m->W2[i];

    loss = (loss + 0.5f * reg_lambda * reg) / (float)N;

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
    // Single parallel region, multiple for's to reduce overhead
#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < L1; i++)
            m->W1[i] -= lr * dW1[i];

#pragma omp for
        for (int i = 0; i < L2; i++)
            m->W2[i] -= lr * dW2[i];

#pragma omp for
        for (int i = 0; i < m->n_hidden; i++)
            m->b1[i] -= lr * db1[i];

#pragma omp for
        for (int i = 0; i < m->n_out; i++)
            m->b2[i] -= lr * db2[i];
    }
#else
    for (int i = 0; i < L1; i++)
        m->W1[i] -= lr * dW1[i];
    for (int i = 0; i < L2; i++)
        m->W2[i] -= lr * dW2[i];
    for (int i = 0; i < m->n_hidden; i++)
        m->b1[i] -= lr * db1[i];
    for (int i = 0; i < m->n_out; i++)
        m->b2[i] -= lr * db2[i];
#endif
}

// --------------------------------------------------------
// Training: backprop + gradient descent (MPI-aware)
// --------------------------------------------------------
void train(
    MLP *m, float *X, int *y, int N,
    float lr0, float reg_lambda,
    int batch_size,
    int num_passes, int print_loss,
    LRSchedule lr_schedule, float k)
{
    if (!m || N <= 0)
        return;

    float lr_t = lr0;
    int world_rank = 0, world_size = 1;
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized)
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
    float *z1 = calloc((size_t)batch_size * m->n_hidden, sizeof(float));
    float *a1 = calloc((size_t)batch_size * m->n_hidden, sizeof(float));
    float *z2 = calloc((size_t)batch_size * m->n_out, sizeof(float));
    float *probs = calloc((size_t)batch_size * m->n_out, sizeof(float));

    float *delta3 = malloc((size_t)batch_size * m->n_out * sizeof(float));
    float *delta2 = malloc((size_t)batch_size * m->n_hidden * sizeof(float));

    float *dW1 = calloc((size_t)L1, sizeof(float));
    float *dW2 = calloc((size_t)L2, sizeof(float));
    float *db1 = calloc((size_t)m->n_hidden, sizeof(float));
    float *db2 = calloc((size_t)m->n_out, sizeof(float));

    float *Xb = malloc((size_t)batch_size * m->n_in * sizeof(float));
    int *yb = malloc((size_t)batch_size * sizeof(int));

    if (!z1 || !a1 || !z2 || !probs || !delta3 || !delta2 ||
        !dW1 || !dW2 || !db1 || !db2 || !Xb || !yb)
    {
        fprintf(stderr, "train: allocation failed (rank %d)\n", world_rank);
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
        return;
    }

    for (int epoch = 0; epoch < num_passes; ++epoch)
    {
        if (lr_schedule != NULL)
            lr_t = lr_schedule(lr0, k, epoch);

        for (int b = 0; b < num_batches; ++b)
        {
            int start = b * batch_size;
            int bs = ((start + batch_size) <= N) ? batch_size : (N - start);

            // --- Copy batch data ---
            for (int i = 0; i < bs; i++)
            {
                int src = start + i;
                yb[i] = y[src];
                memcpy(&Xb[(size_t)i * m->n_in], &X[(size_t)src * m->n_in], (size_t)m->n_in * sizeof(float));
            }

            // --- Forward pass ---
            forward_pass(m, Xb, bs, z1, a1, z2, probs);

            // --- Backward pass ---
            memcpy(delta3, probs, (size_t)bs * m->n_out * sizeof(float));
            for (int i = 0; i < bs; i++)
            {
                int idx = i * m->n_out + yb[i];
                delta3[idx] -= 1.0f;
            }

            memset(dW1, 0, (size_t)L1 * sizeof(float));
            memset(dW2, 0, (size_t)L2 * sizeof(float));
            memset(db1, 0, (size_t)m->n_hidden * sizeof(float));
            memset(db2, 0, (size_t)m->n_out * sizeof(float));

            // Gradients
            matmul_Ta_b(a1, delta3, dW2, bs, m->n_hidden, m->n_out);
            reduce_sum_rows(delta3, db2, bs, m->n_out);

            matmul(delta3, m->W2, delta2, bs, m->n_out, m->n_hidden);

            // activation derivative expects activation value (a1)
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
                for (int i = 0; i < L1; i++)
                    dW1[i] *= inv;
                for (int i = 0; i < L2; i++)
                    dW2[i] *= inv;
                for (int i = 0; i < m->n_hidden; i++)
                    db1[i] *= inv;
                for (int i = 0; i < m->n_out; i++)
                    db2[i] *= inv;
            }

            // Apply update
            update_gradient(m, dW1, dW2, db1, db2, lr_t);
        } // batches

        if (print_loss && (epoch % 100 == 0 || epoch == num_passes - 1))
        {
            int do_print = 1;
            if (mpi_initialized && world_size > 1)
            {
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                do_print = (rank == 0);
            }

            // compute local loss and reduce to global (rank 0)
            float local_L = compute_loss(m, X, y, N, reg_lambda);
            if (!isfinite(local_L))
                local_L = 0.0f; // guard
            float global_L = 0.0f;
            if (mpi_initialized && world_size > 1)
                MPI_Reduce(&local_L, &global_L, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            else
                global_L = local_L;

            if (do_print)
            {
                if (mpi_initialized && world_size > 1)
                    printf("Epoch %d / %d  Loss (sum over ranks): %.6f\n", epoch, num_passes, global_L);
                else
                    printf("Epoch %d / %d  Loss: %.6f\n", epoch, num_passes, global_L);
            }
        }
    } // epochs

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
    if (!m)
        return NULL;
    m->n_in = n_in;
    m->n_hidden = n_hidden;
    m->n_out = n_out;
    m->act = act;

    m->W1 = malloc((size_t)n_in * n_hidden * sizeof(float));
    m->b1 = calloc((size_t)n_hidden, sizeof(float));
    m->W2 = malloc((size_t)n_hidden * n_out * sizeof(float));
    m->b2 = calloc((size_t)n_out, sizeof(float));

    if (!m->W1 || !m->b1 || !m->W2 || !m->b2)
    {
        free(m->W1);
        free(m->b1);
        free(m->W2);
        free(m->b2);
        free(m);
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
