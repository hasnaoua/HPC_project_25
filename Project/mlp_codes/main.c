#include "model.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>  // for timing

int main() {
    // -------------------------
    // Dataset files
    // -------------------------
    const char *file_X = "data/data_X.txt";
    const char *file_y = "data/data_y.txt";

    // -------------------------
    // Network dimensions
    // -------------------------
    int input_dim = 2;       // Number of features
    int output_dim = 2;      // Number of classes
    int hidden_dim = 10;     // Number of hidden neurons

    // -------------------------
    // Count examples
    // -------------------------
    int num_examples = count_lines(file_y);
    printf("Loading %d examples.\n", num_examples);

    // -------------------------
    // Allocate memory for dataset
    // -------------------------
    float *X = malloc(num_examples * input_dim * sizeof(float));
    int   *y = malloc(num_examples * sizeof(int));
    if (!X || !y) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    // -------------------------
    // Load data
    // -------------------------
    load_X(file_X, X, num_examples, input_dim);
    load_y(file_y, y, num_examples);

    // -------------------------
    // Create model
    // -------------------------
    MLP *model = create_model(input_dim, hidden_dim, output_dim, ACT_TANH);

    // -------------------------
    // Train model with timing
    // -------------------------
    printf("Starting training...\n");
    clock_t start_time = clock();

    train(model, X, y, num_examples, 20000, 1); // 20000 iterations, print loss

    clock_t end_time = clock();
    float elapsed_sec = (float)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Training completed in %.2f seconds.\n", elapsed_sec);

    // -------------------------
    // Free memory
    // -------------------------
    free_model(model);
    free(X);
    free(y);

    return 0;
}
