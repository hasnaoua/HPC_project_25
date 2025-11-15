#ifndef MODEL_H
#define MODEL_H

#include "utils.h"
#include <stdint.h>

// =====================================================
// MLP structure
// =====================================================
typedef struct {
    int n_in;
    int n_hidden;
    int n_out;
    Activation act;

    float *W1;
    float *b1;
    float *W2;
    float *b2;
} MLP;

// =====================================================
// Model creation / deletion
// =====================================================
MLP *create_model(int n_in, int n_hidden, int n_out, Activation act);
void free_model(MLP *m);

// =====================================================
// Forward / backward / loss
// =====================================================
void forward_pass(MLP *m, float *X, int N,
                  float *z1, float *a1, float *z2, float *probs);

float compute_loss(MLP *m, float *X, int *y, int N, float reg_lambda);
void update_gradient(MLP *m, float *dW1, float *dW2, float *db1, float *db2, float lr);

// =====================================================
// Training
// =====================================================
void train(MLP *m, float *X, int *y, int N,
           float lr0, float reg_lambda,
           int batch_size, int num_passes, int print_loss,
           LRSchedule lr_schedule, float k);

#endif // MODEL_H
