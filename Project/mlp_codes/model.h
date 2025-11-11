#ifndef MODEL_H
#define MODEL_H

#include "utils.h"

// -------------------------
// Multi-layer Perceptron model
// -------------------------
typedef struct {
    int n_in;       // Input dimension
    int n_hidden;   // Hidden layer size
    int n_out;      // Output dimension

    float *W1, *b1;   // Input -> hidden weights and bias
    float *W2, *b2;   // Hidden -> output weights and bias

    Activation act;   // Activation function for hidden layer
} MLP;

// -------------------------
// Model creation & initialization
// -------------------------
MLP *create_model(int n_in, int n_hidden, int n_out, Activation act);
void free_model(MLP *m);
void init_weights(MLP *m);

// -------------------------
// Forward pass
// -------------------------
void forward_pass(
    MLP *m, float *X, int N,
    float *z1, float *a1, float *z2, float *probs
);

// -------------------------
// Loss computation
// -------------------------
float compute_loss(MLP *m, float *X, int *y, int N, float reg_lambda);

void update_gradient(MLP * m, float *dW1, float *dW2, float *db1, float *db2, float lr);

// -------------------------
// Training (backprop + gradient descent)
// -------------------------
void train(
    MLP *m, float *X, int *y, int N,
    float lr, float reg_lambda,
    int batch_size,
    int num_passes, int print_loss
);

#endif
