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
// Learning rate scheduling
// -------------------------
typedef float (*LRSchedule)(float eta0, float k, int t);

float inverse_time_decay(float eta0, float k, int t);
float exponential_decay(float eta0, float k, int t);

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
void update_gradient(MLP *m, float *dW1, float *dW2, float *db1, float *db2, float lr);

// -------------------------
// Training (backprop + gradient descent)
// -------------------------
void train(MLP *m, 
           float *X_train, int *y_train, int N_train,  float lr0, 
           float reg_lambda, int batch_size, float *X_test, int *y_test,
           int N_test, int num_passes, int print_loss,
           LRSchedule lr_schedule, float k);

// -------------------------
// Evaluation of model
// -------------------------
float evaluate(MLP *m, float *X, int *y, int N);

#endif
