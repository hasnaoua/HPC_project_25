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

// ---------- Thread-safe random ----------

static inline float thread_rand_uniform(unsigned int *state)
{
    return (rand_r(state) + 1.0f) / ((float)RAND_MAX + 2.0f);
}

static inline float thread_randn(unsigned int *state)
{
    float u1 = thread_rand_uniform(state);
    float u2 = thread_rand_uniform(state);
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * (float)M_PI * u2;
    return r * cosf(theta);
}


// ---------- Initialize weights ----------
void init_weights(MLP *m)
{
    if (!m)
        return;

#ifdef _OPENMP
    unsigned int t_seed = (unsigned int)time(NULL);
#pragma omp parallel
    {
        unsigned int seed = t_seed + (unsigned int)omp_get_thread_num() * 9973u;
#pragma omp for
        for (int i = 0; i < m->n_in * m->n_hidden; i++)
            m->W1[i] = thread_randn(&seed) / sqrtf((float)m->n_in);
#pragma omp for
        for (int i = 0; i < m->n_hidden * m->n_out; i++)
            m->W2[i] = thread_randn(&seed) / sqrtf((float)m->n_hidden);
    }
#else
    unsigned int seed = (unsigned int)time(NULL);
    for (int i = 0; i < m->n_in * m->n_hidden; i++)
        m->W1[i] = thread_randn(&seed) / sqrtf((float)m->n_in);
    for (int i = 0; i < m->n_hidden * m->n_out; i++)
        m->W2[i] = thread_randn(&seed) / sqrtf((float)m->n_hidden);
#endif
    memset(m->b1, 0, m->n_hidden * sizeof(float));
    memset(m->b2, 0, m->n_out * sizeof(float));
}

// ---------- Create / free ----------
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

// ---------- Forward pass ----------
void forward_pass(MLP *m, float *X, int N,
                  float *z1, float *a1, float *z2, float *probs)
{

    matmul(X, m->W1, z1, N, m->n_in, m->n_hidden);
    add_bias(z1, m->b1, N, m->n_hidden);

    for (int i = 0; i < N * m->n_hidden; i++)
        a1[i] = m->act.func(z1[i]);

    matmul(a1, m->W2, z2, N, m->n_hidden, m->n_out);
    add_bias(z2, m->b2, N, m->n_out);

    softmax(z2, probs, N, m->n_out);
}

// ---------- Loss ----------
float compute_loss(MLP *m, float *X, int *y, int N, float reg_lambda)
{
    if (N <= 0)
        return 0.0f;

    float loss = 0.0f;

    float *z1   = calloc((size_t)N * m->n_hidden, sizeof(float));
    float *a1   = calloc((size_t)N * m->n_hidden, sizeof(float));
    float *z2   = calloc((size_t)N * m->n_out, sizeof(float));
    float *probs= calloc((size_t)N * m->n_out, sizeof(float));

    if (!z1 || !a1 || !z2 || !probs)
    {
        loss = INFINITY;
        goto cleanup;
    }

    forward_pass(m, X, N, z1, a1, z2, probs);

    // ---------------------------------
    // Parallel loss accumulation
    // ---------------------------------
    #pragma omp parallel for reduction(+:loss)
    for (int i = 0; i < N; i++)
    {
        int cls = y[i];
        float p = probs[i * m->n_out + cls];
        p = fmaxf(p, 1e-12f);
        loss += -logf(p);
    }

    // ---------------------------------
    // Regularization term
    // ---------------------------------
    float reg = 0.0f;

    #pragma omp parallel for reduction(+:reg)
    for (int i = 0; i < m->n_in * m->n_hidden; i++)
        reg += m->W1[i] * m->W1[i];

    #pragma omp parallel for reduction(+:reg)
    for (int i = 0; i < m->n_hidden * m->n_out; i++)
        reg += m->W2[i] * m->W2[i];

    loss = (loss + 0.5f * reg_lambda * reg) / (float)N;

cleanup:
    free(z1);
    free(a1);
    free(z2);
    free(probs);

    return loss;
}


// ---------- Update ----------
void update_gradient(MLP *m, float *dW1, float *dW2, float *db1, float *db2, float lr)
{
    int L1 = m->n_in * m->n_hidden;
    int L2 = m->n_hidden * m->n_out;

#ifdef _OPENMP
#pragma omp parallel for
    for (int i = 0; i < L1; i++)
        m->W1[i] -= lr * dW1[i];
#pragma omp parallel for
    for (int i = 0; i < L2; i++)
        m->W2[i] -= lr * dW2[i];
#pragma omp parallel for
    for (int i = 0; i < m->n_hidden; i++)
        m->b1[i] -= lr * db1[i];
#pragma omp parallel for
    for (int i = 0; i < m->n_out; i++)
        m->b2[i] -= lr * db2[i];
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

// ---------- Training ----------
void train(MLP *m, float *X, int *y, int N,
           float lr0, float reg_lambda,
           int batch_size, int num_passes, int print_loss,
           LRSchedule lr_schedule, float k)
{
    if (!m || N <= 0)
        return;

    int world_rank = 0, world_size = 1;
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    if (batch_size <= 0 || batch_size > N)
        batch_size = N;
    int num_batches = (N + batch_size - 1) / batch_size;
    printf("num batches %d\n", num_batches);


    int L1 = m->n_in * m->n_hidden;
    int L2 = m->n_hidden * m->n_out;

    /* ---- Allocate buffers once ---- */
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
        free(z1); free(a1); free(z2); free(probs);
        free(delta3); free(delta2);
        free(dW1); free(dW2); free(db1); free(db2);
        free(Xb); free(yb);
        return;
    }

    for (int epoch = 0; epoch < num_passes; ++epoch)
    {
        float lr_t = lr_schedule ? lr_schedule(lr0, k, epoch) : lr0;

        for (int b = 0; b < num_batches; ++b)
        {
            int start = b * batch_size;
            int bs = (start + batch_size <= N) ? batch_size : (N - start);

            /*  Copy batch */
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < bs; i++)
            {
                memcpy(&Xb[(size_t)i * m->n_in],
                       &X[(size_t)(start + i) * m->n_in],
                       (size_t)m->n_in * sizeof(float));
                yb[i] = y[start + i];
            }

            /* Forward */
            forward_pass(m, Xb, bs, z1, a1, z2, probs);

            /* delta3 = probs; delta3[i, yb[i]] -= 1 */
            memcpy(delta3, probs, (size_t)bs * m->n_out * sizeof(float));
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < bs; i++)
            {
                int idx = i * m->n_out + yb[i];
                delta3[idx] -= 1.0f;
            }

            /* Zero gradient accumulators */
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < L1; i++) dW1[i] = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < L2; i++) dW2[i] = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < m->n_hidden; i++) db1[i] = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < m->n_out; i++) db2[i] = 0.0f;

            /* Gradients */
            matmul_Ta_b(a1, delta3, dW2, bs, m->n_hidden, m->n_out);
            reduce_sum_rows(delta3, db2, bs, m->n_out);

            matmul(delta3, m->W2, delta2, bs, m->n_out, m->n_hidden);
            
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < bs * m->n_hidden; i++)
                delta2[i] *= m->act.deriv(a1[i]);

            matmul_Ta_b(Xb, delta2, dW1, bs, m->n_in, m->n_hidden);
            reduce_sum_rows(delta2, db1, bs, m->n_hidden);

            /* regularize */
            float inv_bs = 1.0f / (float)bs;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < L1; i++)
                dW1[i] = dW1[i] * inv_bs + reg_lambda * m->W1[i];
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < L2; i++)
                dW2[i] = dW2[i] * inv_bs + reg_lambda * m->W2[i];
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < m->n_hidden; i++)
                db1[i] *= inv_bs;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < m->n_out; i++)
                db2[i] *= inv_bs;

            /* MPI gradient reduction */
            if (mpi_initialized && world_size > 1)
            {
                MPI_Allreduce(MPI_IN_PLACE, dW1, L1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, dW2, L2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, db1, m->n_hidden, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, db2, m->n_out, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

                float inv = 1.0f / (float)world_size;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                for (int i = 0; i < L1; i++) dW1[i] *= inv;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                for (int i = 0; i < L2; i++) dW2[i] *= inv;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                for (int i = 0; i < m->n_hidden; i++) db1[i] *= inv;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                for (int i = 0; i < m->n_out; i++) db2[i] *= inv;
            }

            /* update */
            update_gradient(m, dW1, dW2, db1, db2, lr_t);
        } 

        /* Print loss */
        if (print_loss && (epoch % 100 == 0 || epoch == num_passes - 1))
        {
            float local_L = compute_loss(m, X, y, N, reg_lambda);
            float global_L = local_L;
            if (mpi_initialized && world_size > 1)
            {
                MPI_Reduce(&local_L, &global_L, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
                if (world_rank == 0)
                    global_L /= (float)world_size;
            }
            if (world_rank == 0)
            {
                printf("Epoch %d / %d  Loss: %.6f\n", epoch, num_passes, global_L);
            }
        }
    }

    /* ---- Free buffers ---- */
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





