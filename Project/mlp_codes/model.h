#ifndef MODEL_H
#define MODEL_H

// ----------------------
// Global variables
// ----------------------
extern int num_examples;
extern int nn_input_dim;
extern int nn_output_dim;

extern float reg_lambda;
extern float epsilon;

// Dataset
extern float *X;
extern int *y;

// ----------------------
// Functions
// ----------------------
float calculate_loss(float *W1, float *b1,
                     float *W2, float *b2,
                     int nn_hdim);

void build_model(int nn_hdim,
                 int num_passes,
                 int print_loss);

#endif
