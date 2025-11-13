#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// --------------------------------------------------------
// Learning Rate Schedules
// --------------------------------------------------------
float inverse_time_decay(float eta0, float k, int t) {
    return eta0 / (1.0f + k * t);
}

float exponential_decay(float eta0, float k, int t) {
    return eta0 * expf(-k * t);
}

// --------------------------------------------------------
// Weight initialization
// --------------------------------------------------------
void init_weights(MLP *m) {
    for (int i = 0; i < m->n_in * m->n_hidden; i++)
        m->W1[i] = randn() / sqrtf(m->n_in);

    for (int i = 0; i < m->n_hidden * m->n_out; i++)
        m->W2[i] = randn() / sqrtf(m->n_hidden);

    memset(m->b1, 0, m->n_hidden * sizeof(float));
    memset(m->b2, 0, m->n_out * sizeof(float));
}

// --------------------------------------------------------
// Forward pass
// --------------------------------------------------------
void forward_pass(MLP *m, float *X, int N,
                  float *z1, float *a1, float *z2, float *probs) {
    matmul(X, m->W1, z1, N, m->n_in, m->n_hidden);
    add_bias(z1, m->b1, N, m->n_hidden);

    for (int i = 0; i < N * m->n_hidden; i++)
        a1[i] = m->act.func(z1[i]);

    matmul(a1, m->W2, z2, N, m->n_hidden, m->n_out);
    add_bias(z2, m->b2, N, m->n_out);

    softmax(z2, probs, N, m->n_out);
}

// --------------------------------------------------------
// Compute loss (cross-entropy + L2 regularization)
// --------------------------------------------------------
float compute_loss(MLP *m, float *X, int *y, int N, float reg_lambda) {
    float *z1 = calloc(N * m->n_hidden, sizeof(float));
    float *a1 = calloc(N * m->n_hidden, sizeof(float));
    float *z2 = calloc(N * m->n_out, sizeof(float));
    float *probs = calloc(N * m->n_out, sizeof(float));

    forward_pass(m, X, N, z1, a1, z2, probs);

    float loss = 0.0f;
    for (int i = 0; i < N; i++)
        loss += -logf(probs[i * m->n_out + y[i]]);

    float reg = 0.0f;
    for (int i = 0; i < m->n_in * m->n_hidden; i++) reg += m->W1[i] * m->W1[i];
    for (int i = 0; i < m->n_hidden * m->n_out; i++) reg += m->W2[i] * m->W2[i];

    loss = (loss + 0.5f * m->reg_lambda * reg) / N;

    free(z1); free(a1); free(z2); free(probs);
    return loss;
}

// --------------------------------------------------------
// Gradient update
// --------------------------------------------------------
void update_gradient(MLP *m, float *dW1, float *dW2, float *db1, float *db2, float lr) {
    for (int i = 0; i < m->n_in * m->n_hidden; i++)
        m->W1[i] -= lr * dW1[i];
    for (int i = 0; i < m->n_hidden; i++)
        m->b1[i] -= lr * db1[i];
    for (int i = 0; i < m->n_hidden * m->n_out; i++)
        m->W2[i] -= lr * dW2[i];
    for (int i = 0; i < m->n_out; i++)
        m->b2[i] -= lr * db2[i];
}

// --------------------------------------------------------
// Training: backprop + dynamic LR
// --------------------------------------------------------
void train(MLP *m,
           float *X_train, int *y_train, int N_train, float lr0,
           float reg_lambda, int batch_size, float *X_test, int *y_test,
           int N_test, int num_passes, int print_loss,
           LRSchedule lr_schedule, float k)
{
    const char *act_name = "unknown";
    if (m->act.func == ACT_TANH.func) act_name = "tanh";
    else if (m->act.func == ACT_RELU.func) act_name = "relu";
    else if (m->act.func == ACT_LEAKY_RELU.func) act_name = "leak relu";
    else if (m->act.func == ACT_SIGMOID.func) act_name = "sigmoid";
    

    char loss_file[128];
    sprintf(loss_file, "Losses/loss_%s.txt", act_name);
    FILE *f = fopen(loss_file, "w");
    if (!f) { perror("open loss file"); return; }

    if (batch_size <= 0) batch_size = N_train;
    if (batch_size > N_train) batch_size = N_train;

    int maxB = batch_size;
    float *z1 = calloc(maxB * m->n_hidden, sizeof(float));
    float *a1 = calloc(maxB * m->n_hidden, sizeof(float));
    float *z2 = calloc(maxB * m->n_out, sizeof(float));
    float *probs = calloc(maxB * m->n_out, sizeof(float));
    float *delta3 = malloc(maxB * m->n_out * sizeof(float));
    float *delta2 = malloc(maxB * m->n_hidden * sizeof(float));
    float *dW1 = malloc(m->n_in * m->n_hidden * sizeof(float));
    float *dW2 = malloc(m->n_hidden * m->n_out * sizeof(float));
    float *db1 = malloc(m->n_hidden * sizeof(float));
    float *db2 = malloc(m->n_out * sizeof(float));
    float *X_batch = malloc(maxB * m->n_in * sizeof(float));
    int *y_batch = malloc(maxB * sizeof(int));

    int num_batches = (N_train + batch_size - 1) / batch_size;

    for (int epoch = 0; epoch < num_passes; ++epoch) {
        float lr_t = (lr_schedule != NULL) ? lr_schedule(lr0, k, epoch) : lr0;
        int cursor = 0;

        for (int b = 0; b < num_batches; ++b) {
            int bs = (cursor + batch_size <= N_train) ? batch_size : (N_train - cursor);
            for (int i = 0; i < bs; ++i) {
                int src = cursor + i;
                y_batch[i] = y_train[src];
                for (int f = 0; f < m->n_in; ++f)
                    X_batch[i * m->n_in + f] = X_train[src * m->n_in + f];
            }
            cursor += bs;

            forward_pass(m, X_batch, bs, z1, a1, z2, probs);
            memcpy(delta3, probs, bs * m->n_out * sizeof(float));
            for (int i = 0; i < bs; ++i)
                delta3[i * m->n_out + y_batch[i]] -= 1.0f;

            memset(dW1, 0, m->n_in * m->n_hidden * sizeof(float));
            memset(dW2, 0, m->n_hidden * m->n_out * sizeof(float));
            memset(db1, 0, m->n_hidden * sizeof(float));
            memset(db2, 0, m->n_out * sizeof(float));

            matmul_Ta_b(a1, delta3, dW2, bs, m->n_hidden, m->n_out);
            reduce_sum_rows(delta3, db2, bs, m->n_out);
            matmul(delta3, m->W2, delta2, bs, m->n_out, m->n_hidden);
            for (int i = 0; i < bs * m->n_hidden; ++i)
                delta2[i] *= m->act.deriv(a1[i]);

            matmul_Ta_b(X_batch, delta2, dW1, bs, m->n_in, m->n_hidden);
            reduce_sum_rows(delta2, db1, bs, m->n_hidden);

            float inv_bs = 1.0f / (float)bs;
            for (int i = 0; i < m->n_in * m->n_hidden; ++i) dW1[i] = dW1[i] * inv_bs + reg_lambda * m->W1[i];
            for (int i = 0; i < m->n_hidden * m->n_out; ++i) dW2[i] = dW2[i] * inv_bs + reg_lambda * m->W2[i];
            for (int i = 0; i < m->n_hidden; ++i) db1[i] *= inv_bs;
            for (int i = 0; i < m->n_out; ++i) db2[i] *= inv_bs;

            update_gradient(m, dW1, dW2, db1, db2, lr_t);
        }

        if (print_loss && (epoch % 10 == 0 || epoch == num_passes - 1)) {
            float train_loss = compute_loss(m, X_train, y_train, N_train, reg_lambda);
            float eval_loss  = compute_loss(m, X_test, y_test, N_test, reg_lambda);
            fprintf(f, "%d %.6f %.6f\n", epoch, train_loss, eval_loss);
            printf("[%s] Epoch %d | LR = %.6f | Train Loss = %.6f | Eval Loss = %.6f\n",
                   act_name, epoch, lr_t, train_loss, eval_loss);
        }
    }

    fclose(f);
    free(z1); free(a1); free(z2); free(probs);
    free(delta3); free(delta2); free(dW1); free(dW2);
    free(db1); free(db2); free(X_batch); free(y_batch);
}

// --------------------------------------------------------
// Evaluate
// --------------------------------------------------------
float evaluate(MLP *m, float *X, int *y, int N) {
    int correct = 0;
    float *z1 = calloc(N * m->n_hidden, sizeof(float));
    float *a1 = calloc(N * m->n_hidden, sizeof(float));
    float *z2 = calloc(N * m->n_out, sizeof(float));
    float *probs = calloc(N * m->n_out, sizeof(float));

    forward_pass(m, X, N, z1, a1, z2, probs);
    for (int i = 0; i < N; i++) {
        int pred = 0;
        float maxp = probs[i * m->n_out];
        for (int j = 1; j < m->n_out; j++) {
            if (probs[i * m->n_out + j] > maxp) {
                maxp = probs[i * m->n_out + j];
                pred = j;
            }
        }
        if (pred == y[i]) correct++;
    }

    free(z1); free(a1); free(z2); free(probs);
    return (float)correct / N;
}

// --------------------------------------------------------
// Create and free model
// --------------------------------------------------------
MLP *create_model(int n_in, int n_hidden, int n_out, Activation act) {
    MLP *m = malloc(sizeof(MLP));
    m->n_in = n_in;
    m->n_hidden = n_hidden;
    m->n_out = n_out;
    m->reg_lambda = 0.01f;
    m->lr = 0.01f;
    m->act = act;

    m->W1 = malloc(n_in * n_hidden * sizeof(float));
    m->b1 = calloc(n_hidden, sizeof(float));
    m->W2 = malloc(n_hidden * n_out * sizeof(float));
    m->b2 = calloc(n_out, sizeof(float));

    init_weights(m);
    return m;
}

void free_model(MLP *m) {
    if (!m) return;
    free(m->W1); free(m->b1); free(m->W2); free(m->b2);
    free(m);
}
