#include "model.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// --------------------------
// Global variables
// --------------------------
int num_examples;
int nn_input_dim;
int nn_output_dim;
float reg_lambda = 0.01f;
float epsilon = 0.01f;

float *X;  // (num_examples × nn_input_dim)
int   *y;  // labels


// ============================================================
// 1. Loss function (float, optimized)
// ============================================================
float calculate_loss(float *W1, float *b1, float *W2, float *b2, int nn_hdim)
{
    float *z1   = calloc(num_examples * nn_hdim, sizeof(float));
    float *a1   = calloc(num_examples * nn_hdim, sizeof(float));
    float *z2   = calloc(num_examples * nn_output_dim, sizeof(float));
    float *probs = calloc(num_examples * nn_output_dim, sizeof(float));

    // Forward
    matmul(X, W1, z1, num_examples, nn_input_dim, nn_hdim);
    add_bias(z1, b1, num_examples, nn_hdim);

    for (int i = 0; i < num_examples * nn_hdim; i++)
        a1[i] = tanhf(z1[i]);

    matmul(a1, W2, z2, num_examples, nn_hdim, nn_output_dim);
    add_bias(z2, b2, num_examples, nn_output_dim);

    softmax(z2, probs, num_examples, nn_output_dim);

    // Cross-entropy
    float data_loss = 0.0f;
    for (int i = 0; i < num_examples; i++)
        data_loss += -logf(probs[i * nn_output_dim + y[i]]);

    // Regularization
    float reg = 0.0f;
    for (int i = 0; i < nn_input_dim * nn_hdim; i++)
        reg += W1[i] * W1[i];
    for (int i = 0; i < nn_hdim * nn_output_dim; i++)
        reg += W2[i] * W2[i];

    data_loss += reg_lambda * 0.5f * reg;

    float loss = data_loss / num_examples;

    free(z1);
    free(a1);
    free(z2);
    free(probs);

    return loss;
}



// ============================================================
// 2. build_model (optimized float version)
// ============================================================
void build_model(int nn_hdim, int num_passes, int print_loss)
{
    clock_t start = clock();
    srand(0);

    // Allocate weights
    float *W1 = malloc(nn_input_dim * nn_hdim * sizeof(float));
    float *b1 = calloc(nn_hdim, sizeof(float));
    float *W2 = malloc(nn_hdim * nn_output_dim * sizeof(float));
    float *b2 = calloc(nn_output_dim, sizeof(float));

    // init with Xavier
    for (int i = 0; i < nn_input_dim * nn_hdim; i++)
        W1[i] = randn() / sqrtf(nn_input_dim);
    for (int i = 0; i < nn_hdim * nn_output_dim; i++)
        W2[i] = randn() / sqrtf(nn_hdim);

    // Temporary buffers (allocated once)
    float *z1     = calloc(num_examples * nn_hdim, sizeof(float));
    float *a1     = calloc(num_examples * nn_hdim, sizeof(float));
    float *dtanh  = malloc(num_examples * nn_hdim * sizeof(float));
    float *z2     = calloc(num_examples * nn_output_dim, sizeof(float));
    float *probs  = calloc(num_examples * nn_output_dim, sizeof(float));
    float *delta3 = malloc(num_examples * nn_output_dim * sizeof(float));
    float *delta2 = malloc(num_examples * nn_hdim * sizeof(float));
    float *dW1    = malloc(nn_input_dim * nn_hdim * sizeof(float));
    float *dW2    = malloc(nn_hdim * nn_output_dim * sizeof(float));
    float *db1    = malloc(nn_hdim * sizeof(float));
    float *db2    = malloc(nn_output_dim * sizeof(float));



    // ======================================
    // Training loop
    // ======================================
    for (int it = 0; it < num_passes; it++)
    {
        // ----------------------
        // Forward propagation
        // ----------------------
        matmul(X, W1, z1, num_examples, nn_input_dim, nn_hdim);
        add_bias(z1, b1, num_examples, nn_hdim);

        for (int i = 0; i < num_examples * nn_hdim; i++) {
            float a = tanhf(z1[i]);
            a1[i] = a;
            dtanh[i] = 1.0f - (a * a);
        }

        matmul(a1, W2, z2, num_examples, nn_hdim, nn_output_dim);
        add_bias(z2, b2, num_examples, nn_output_dim);
        softmax(z2, probs, num_examples, nn_output_dim);


        // ----------------------
        // Zero gradients
        // ----------------------
        memset(dW1, 0, nn_input_dim * nn_hdim * sizeof(float));
        memset(dW2, 0, nn_hdim * nn_output_dim * sizeof(float));
        memset(db1, 0, nn_hdim * sizeof(float));
        memset(db2, 0, nn_output_dim * sizeof(float));


        // ----------------------
        // Backpropagation
        // ----------------------

        // delta3 = probs - one_hot
        for (int i = 0; i < num_examples * nn_output_dim; i++)
            delta3[i] = probs[i];

        for (int i = 0; i < num_examples; i++)
            delta3[i * nn_output_dim + y[i]] -= 1.0f;


        // dW2 = a1.T * delta3
        for (int j = 0; j < nn_hdim; j++)
            for (int k = 0; k < nn_output_dim; k++)
            {
                float sum = 0.0f;
                for (int n = 0; n < num_examples; n++)
                    sum += a1[n * nn_hdim + j] * delta3[n * nn_output_dim + k];
                dW2[j * nn_output_dim + k] = sum;
            }

        // db2
        for (int k = 0; k < nn_output_dim; k++)
        {
            float sum = 0.0f;
            for (int n = 0; n < num_examples; n++)
                sum += delta3[n * nn_output_dim + k];
            db2[k] = sum;
        }

        // delta2 = delta3 * W2.T .* dtanh
        for (int n = 0; n < num_examples; n++)
        {
            int base_nh = n * nn_hdim;
            int base_no = n * nn_output_dim;

            for (int j = 0; j < nn_hdim; j++)
            {
                float sum = 0.0f;
                for (int k = 0; k < nn_output_dim; k++)
                    sum += delta3[base_no + k] * W2[j * nn_output_dim + k];

                delta2[base_nh + j] = sum * dtanh[base_nh + j];
            }
        }

        // dW1 = X.T * delta2
        for (int i = 0; i < nn_input_dim; i++)
            for (int j = 0; j < nn_hdim; j++)
            {
                float sum = 0.0f;
                for (int n = 0; n < num_examples; n++)
                    sum += X[n * nn_input_dim + i] * delta2[n * nn_hdim + j];
                dW1[i * nn_hdim + j] = sum;
            }

        // db1
        for (int j = 0; j < nn_hdim; j++)
        {
            float sum = 0.0f;
            for (int n = 0; n < num_examples; n++)
                sum += delta2[n * nn_hdim + j];
            db1[j] = sum;
        }

        // Regularization
        for (int i = 0; i < nn_hdim * nn_output_dim; i++)
            dW2[i] += reg_lambda * W2[i];
        for (int i = 0; i < nn_input_dim * nn_hdim; i++)
            dW1[i] += reg_lambda * W1[i];


        // ----------------------
        // Gradient descent update
        // ----------------------
        for (int i = 0; i < nn_input_dim * nn_hdim; i++)
            W1[i] -= epsilon * dW1[i];

        for (int i = 0; i < nn_hdim; i++)
            b1[i] -= epsilon * db1[i];

        for (int i = 0; i < nn_hdim * nn_output_dim; i++)
            W2[i] -= epsilon * dW2[i];

        for (int i = 0; i < nn_output_dim; i++)
            b2[i] -= epsilon * db2[i];


        // print loss
        if (print_loss && it % 1000 == 0)
            printf("Loss after %d iterations: %.6f\n",
                   it, calculate_loss(W1, b1, W2, b2, nn_hdim));
    }



    // Free backprop buffers
    free(z1); free(a1); free(dtanh);
    free(z2); free(probs);
    free(delta3); free(delta2);
    free(dW1); free(dW2);
    free(db1); free(db2);


    // Save weights
    FILE *fw1 = fopen("output/W1.txt", "w");
    FILE *fb1 = fopen("output/b1.txt", "w");
    FILE *fw2 = fopen("output/W2.txt", "w");
    FILE *fb2 = fopen("output/b2.txt", "w");

    for (int i = 0; i < nn_input_dim * nn_hdim; i++)
        fprintf(fw1, "%f\n", W1[i]);
    for (int i = 0; i < nn_hdim; i++)
        fprintf(fb1, "%f\n", b1[i]);
    for (int i = 0; i < nn_hdim * nn_output_dim; i++)
        fprintf(fw2, "%f\n", W2[i]);
    for (int i = 0; i < nn_output_dim; i++)
        fprintf(fb2, "%f\n", b2[i]);

    fclose(fw1);
    fclose(fb1);
    fclose(fw2);
    fclose(fb2);

    free(W1); free(W2);
    free(b1); free(b2);

    float seconds = (float)(clock() - start) / CLOCKS_PER_SEC;
    printf("Execution time: %f s\n", seconds);
}
