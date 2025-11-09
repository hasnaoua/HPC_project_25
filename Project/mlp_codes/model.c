#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
void forward_pass(
    MLP *m, float *X, int N,
    float *z1, float *a1, float *z2, float *probs
) {
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
// Compute loss (cross-entropy + L2 regularization)
// --------------------------------------------------------
float compute_loss(MLP *m, float *X, int *y, int N) {
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
    for (int i = 0; i < m->n_in * m->n_hidden; i++) reg += m->W1[i] * m->W1[i];
    for (int i = 0; i < m->n_hidden * m->n_out; i++) reg += m->W2[i] * m->W2[i];

    loss = (loss + 0.5f * m->reg_lambda * reg) / N;

    free(z1); free(a1); free(z2); free(probs);
    return loss;
}

// --------------------------------------------------------
// Training: backprop + gradient descent
// --------------------------------------------------------
void train(
    MLP *m, float *X, int *y, int N,
    int num_passes, int print_loss
) {
    // Allocate forward buffers
    float *z1 = calloc(N * m->n_hidden, sizeof(float));
    float *a1 = calloc(N * m->n_hidden, sizeof(float));
    float *z2 = calloc(N * m->n_out, sizeof(float));
    float *probs = calloc(N * m->n_out, sizeof(float));

    // Gradients
    float *delta3 = malloc(N * m->n_out * sizeof(float));
    float *delta2 = malloc(N * m->n_hidden * sizeof(float));
    float *dW1 = malloc(m->n_in * m->n_hidden * sizeof(float));
    float *dW2 = malloc(m->n_hidden * m->n_out * sizeof(float));
    float *db1 = malloc(m->n_hidden * sizeof(float));
    float *db2 = malloc(m->n_out * sizeof(float));

    for (int it = 0; it < num_passes; it++) {
        forward_pass(m, X, N, z1, a1, z2, probs);

        // delta3 = probs - one_hot
        memcpy(delta3, probs, N * m->n_out * sizeof(float));
        for (int i = 0; i < N; i++)
            delta3[i * m->n_out + y[i]] -= 1.0f;

        // Gradients for W2, b2
        matmul_Ta_b(a1, delta3, dW2, N, m->n_hidden, m->n_out);
        reduce_sum_rows(delta3, db2, N, m->n_out);

        // Backprop to hidden layer
        matmul(delta3, m->W2, delta2, N, m->n_out, m->n_hidden);
        for (int i = 0; i < N * m->n_hidden; i++)
            delta2[i] *= m->act.deriv(a1[i]);

        // Gradients for W1, b1
        matmul_Ta_b(X, delta2, dW1, N, m->n_in, m->n_hidden);
        reduce_sum_rows(delta2, db1, N, m->n_hidden);

        // Regularization
        for (int i = 0; i < m->n_in * m->n_hidden; i++) dW1[i] += m->reg_lambda * m->W1[i];
        for (int i = 0; i < m->n_hidden * m->n_out; i++) dW2[i] += m->reg_lambda * m->W2[i];

        // Gradient descent update
        for (int i = 0; i < m->n_in * m->n_hidden; i++) m->W1[i] -= m->lr * dW1[i];
        for (int i = 0; i < m->n_hidden; i++) m->b1[i] -= m->lr * db1[i];
        for (int i = 0; i < m->n_hidden * m->n_out; i++) m->W2[i] -= m->lr * dW2[i];
        for (int i = 0; i < m->n_out; i++) m->b2[i] -= m->lr * db2[i];

        if (print_loss && it % 1000 == 0)
            printf("Loss %d: %.4f\n", it, compute_loss(m, X, y, N));
    }

    free(z1); free(a1); free(z2); free(probs);
    free(delta3); free(delta2);
    free(dW1); free(dW2);
    free(db1); free(db2);
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
    free(m->W1);
    free(m->b1);
    free(m->W2);
    free(m->b2);
    free(m);
}
