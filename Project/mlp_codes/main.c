#include "model.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>   // pour mkdir
#include <string.h>

int main() {
    srand(time(NULL));

    // -------------------------
    // Dataset files
    // -------------------------
    const char *file_X = "data/data_X.txt";
    const char *file_y = "data/data_y.txt";

    // -------------------------
    // Network dimensions
    // -------------------------
    int input_dim  = 2;     // nombre de features
    int hidden_dim = 10;    // nombre de neurones cachés
    int output_dim = 2;     // nombre de classes

    // -------------------------
    // Load data
    // -------------------------
    int num_examples = count_lines(file_y);
    printf("Loading %d examples.\n", num_examples);

    float *X = malloc(num_examples * input_dim * sizeof(float));
    int   *y = malloc(num_examples * sizeof(int));

    if (!X || !y) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    load_X(file_X, X, num_examples, input_dim);
    load_y(file_y, y, num_examples);

    // -------------------------
    // Split into training / testing
    // -------------------------
    int train_size = (int)(0.8 * num_examples);
    int test_size  = num_examples - train_size;

    float *X_train = X;
    int   *y_train = y;

    float *X_test = X + train_size * input_dim;
    int   *y_test = y + train_size;

    printf("Training on %d samples, testing on %d samples.\n", train_size, test_size);

    // -------------------------
    // Create directory for losses (optional, safety)
    // -------------------------
    mkdir("Losses", 0777);

    // -------------------------
    // Train for each activation
    // -------------------------
    Activation activations[] = {ACT_TANH, ACT_RELU, ACT_SIGMOID};
    const char *act_names[]  = {"tanh", "relu", "sigmoid"};
    int num_acts = 3;

    for (int i = 0; i < num_acts; i++) {
        printf("\n========================================\n");
        printf("Training with activation: %s\n", act_names[i]);
        printf("========================================\n");

        MLP *model = create_model(input_dim, hidden_dim, output_dim, activations[i]);

        clock_t start_time = clock();
        train(model, X_train, y_train, train_size, 20000, 1);
        clock_t end_time = clock();

        float elapsed = (float)(end_time - start_time) / CLOCKS_PER_SEC;
        printf("Training (%s) completed in %.2f seconds.\n", act_names[i], elapsed);

        free_model(model);
    }

    // -------------------------
    // Cleanup
    // -------------------------
    free(X);
    free(y);

    printf("\nAll training runs completed. Losses saved in ./Losses/\n");
    printf("You can now run the Python script to plot them.\n");

    return 0;
}
