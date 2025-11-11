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
    float reg_lambda; // Regularization strength
    float lr;         // Learning rate

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
float compute_loss(MLP *m, float *X, int *y, int N);

// -------------------------
// Training (backprop + gradient descent)
// -------------------------
void train(MLP *m,
           float *X_train, int *y_train, int N_train,
           float *X_test,  int *y_test,  int N_test,
           int num_passes, int print_loss);

// -------------------------
// Evaluation of model
// -------------------------
float evaluate(MLP *m, float *X, int *y, int N);


#endif
