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

    loss = (loss + 0.5f * m->lr * reg) / N; // Compute average loss

    free(z1); free(a1); free(z2); free(probs);
    return loss;
}
// --------------------------------------------------------
// Training: backprop + gradient descent + eval loss logging
// --------------------------------------------------------
void train(MLP *m, 
           float *X_train, int *y_train, int N_train,
           float *X_test,  int *y_test,  int N_test,
           int num_passes, int print_loss)
{
    //system("mkdir -p Losses");

    // Identify activation function for file naming
    const char *act_name = "unknown";
    if (m->act.func == ACT_TANH.func) act_name = "tanh";
    else if (m->act.func == ACT_RELU.func) act_name = "relu";
    else if (m->act.func == ACT_SIGMOID.func) act_name = "sigmoid";

    // Output file for train + eval loss
    char loss_file[128];
    sprintf(loss_file, "Losses/loss_%s.txt", act_name);
    FILE *f = fopen(loss_file, "w");
    if (!f) {
        perror("open loss file");
        return;
    }

    // Buffers
    float *z1 = calloc(N_train * m->n_hidden, sizeof(float));
    float *a1 = calloc(N_train * m->n_hidden, sizeof(float));
    float *z2 = calloc(N_train * m->n_out, sizeof(float));
    float *probs = calloc(N_train * m->n_out, sizeof(float));

    float *delta3 = malloc(N_train * m->n_out * sizeof(float));
    float *delta2 = malloc(N_train * m->n_hidden * sizeof(float));
    float *dW1 = malloc(m->n_in * m->n_hidden * sizeof(float));
    float *dW2 = malloc(m->n_hidden * m->n_out * sizeof(float));
    float *db1 = malloc(m->n_hidden * sizeof(float));
    float *db2 = malloc(m->n_out * sizeof(float));

    // Training loop
    for (int it = 0; it < num_passes; it++) {
        forward_pass(m, X_train, N_train, z1, a1, z2, probs);

        // delta3 = probs - one_hot(y)
        memcpy(delta3, probs, N_train * m->n_out * sizeof(float));
        for (int i = 0; i < N_train; i++)
            delta3[i * m->n_out + y_train[i]] -= 1.0f;

        // Gradients
        matmul_Ta_b(a1, delta3, dW2, N_train, m->n_hidden, m->n_out);
        reduce_sum_rows(delta3, db2, N_train, m->n_out);

        matmul(delta3, m->W2, delta2, N_train, m->n_out, m->n_hidden);
        for (int i = 0; i < N_train * m->n_hidden; i++)
            delta2[i] *= m->act.deriv(a1[i]);

        matmul_Ta_b(X_train, delta2, dW1, N_train, m->n_in, m->n_hidden);
        reduce_sum_rows(delta2, db1, N_train, m->n_hidden);

        // Regularization
        for (int i = 0; i < m->n_in * m->n_hidden; i++) dW1[i] += m->reg_lambda * m->W1[i];
        for (int i = 0; i < m->n_hidden * m->n_out; i++) dW2[i] += m->reg_lambda * m->W2[i];

        // Gradient descent
        for (int i = 0; i < m->n_in * m->n_hidden; i++) m->W1[i] -= m->lr * dW1[i];
        for (int i = 0; i < m->n_hidden; i++) m->b1[i] -= m->lr * db1[i];
        for (int i = 0; i < m->n_hidden * m->n_out; i++) m->W2[i] -= m->lr * dW2[i];
        for (int i = 0; i < m->n_out; i++) m->b2[i] -= m->lr * db2[i];

        // Record losses every 1000 iters
        if (it % 1000 == 0) {
            float train_loss = compute_loss(m, X_train, y_train, N_train);
            float eval_loss  = compute_loss(m, X_test, y_test, N_test);

            fprintf(f, "%d %.6f %.6f\n", it, train_loss, eval_loss);

            if (print_loss)
                printf("[%s] Iter %d: train_loss=%.6f, eval_loss=%.6f\n",
                       act_name, it, train_loss, eval_loss);
        }
    }

    // Cleanup
    fclose(f);
    free(z1); free(a1); free(z2); free(probs);
    free(delta3); free(delta2);
    free(dW1); free(dW2);
    free(db1); free(db2);
}


// --------------------------------------------------------
// Evaluate model accuracy and loss on test data
// --------------------------------------------------------
float evaluate(MLP *m, float *X, int *y, int N)
{
    int correct = 0;

    float *z1 = calloc(N * m->n_hidden, sizeof(float));
    float *a1 = calloc(N * m->n_hidden, sizeof(float));
    float *z2 = calloc(N * m->n_out, sizeof(float));
    float *probs = calloc(N * m->n_out, sizeof(float));

    // Forward pass
    forward_pass(m, X, N, z1, a1, z2, probs);

    // Compute accuracy
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

    float acc = (float)correct / N;

    // Cleanup
    free(z1); free(a1); free(z2); free(probs);

    return acc;
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
