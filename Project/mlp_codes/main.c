#include "model.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <string.h>

int main() {
    srand(time(NULL));

    const char *file_X = "data/data_X.txt";
    const char *file_y = "data/data_y.txt";

    int input_dim = 2;
    int output_dim = 2;
    int hidden_dim = 10;
    float reg_lambda = 0.01f;
    float lr0 = 0.01f;
    int batch_size = 64;
    int num_passes = 2000;
    float decay_k = 0.001f;

    int num_examples = count_lines(file_y);
    printf("Loading %d examples.\n", num_examples);

    float *X = malloc(num_examples * input_dim * sizeof(float));
    int *y = malloc(num_examples * sizeof(int));
    load_X(file_X, X, num_examples, input_dim);
    load_y(file_y, y, num_examples);

    int train_size = (int)(0.7 * num_examples);
    int test_size = num_examples - train_size;

    float *X_train = X;
    int *y_train = y;
    float *X_test = X + train_size * input_dim;
    int *y_test = y + train_size;

    mkdir("Losses", 0777);

    Activation activations[] = {ACT_TANH, ACT_RELU, ACT_SIGMOID};
    const char *act_names[]  = {"tanh", "relu", "sigmoid"};

    // Learning rate mode: 0=fixed, 1=inverse, 2=exponential
    int lr_choice = 2;
    LRSchedule schedule = NULL;
    if (lr_choice == 1) schedule = inverse_time_decay;
    else if (lr_choice == 2) schedule = exponential_decay;

    // Loop over activation functions

    for (int i = 0; i < 3; i++) {
        printf("\n========================================\n");
        printf("Training with activation: %s\n", act_names[i]);
        printf("========================================\n");

        MLP *model = create_model(input_dim, hidden_dim, output_dim, activations[i]);

        clock_t start = clock();
        train(model, X_train, y_train, train_size, lr0,
              reg_lambda, batch_size, X_test, y_test,
              test_size, num_passes, 1, schedule, decay_k);
        clock_t end = clock();

        printf("Training completed in %.2f s\n", (float)(end - start) / CLOCKS_PER_SEC);
        float acc = evaluate(model, X_test, y_test, test_size);
        printf("Test accuracy (%s): %.2f%%\n", act_names[i], acc * 100);

        free_model(model);
    }

    free(X);
    free(y);
    return 0;
}
