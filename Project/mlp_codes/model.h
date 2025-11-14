#ifndef MODEL_H
#define MODEL_H

#include "utils.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

// ----------------------------------------------------
// Multi-layer Perceptron (MLP) structure
// ----------------------------------------------------
typedef struct
{
    int n_in;      // Input dimension
    int n_hidden;  // Hidden-layer width
    int n_out;     // Output dimension

    // Parameters
    float *W1, *b1;  // Input → hidden
    float *W2, *b2;  // Hidden → output

    Activation act;  // Activation for hidden layer
} MLP;

// ----------------------------------------------------
// Model creation / destruction / initialization
// ----------------------------------------------------
MLP  *create_model(int n_in, int n_hidden, int n_out, Activation act);
void  free_model(MLP *m);
void  init_weights(MLP *m);

// ----------------------------------------------------
// Forward propagation
// ----------------------------------------------------
void forward_pass(
    MLP *m,
    float *X,      // [N × n_in]
    int   N,
    float *z1,     // [N × n_hidden]
    float *a1,     // [N × n_hidden]
    float *z2,     // [N × n_out]
    float *probs   // [N × n_out] (softmax output)
);

// ----------------------------------------------------
// Loss computation (cross entropy + L2 regularization)
// ----------------------------------------------------
float compute_loss(
    MLP *m,
    float *X,
    int   *y,
    int    N,
    float  reg_lambda
);

// ----------------------------------------------------
// Gradient descent parameter update
// ----------------------------------------------------
void update_gradient(
    MLP   *m,
    float *dW1, float *dW2,
    float *db1, float *db2,
    float  lr
);

// ----------------------------------------------------
// Training loop (MPI aware)
// ----------------------------------------------------
void train(
    MLP *m,
    float *X, int *y, int N,
    float lr_init,
    float reg_lambda,
    int batch_size,
    int num_passes,
    int print_loss,
    LRSchedule lr_schedule,  // NULL or function pointer
    float decay_k
);

#endif // MODEL_H
