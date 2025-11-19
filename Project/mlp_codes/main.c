#include "model.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // ----------------------------------------
    // Dataset paths
    // ----------------------------------------
    const char *file_X = "data/data_X.txt";
    const char *file_y = "data/data_y.txt";

    // ----------------------------------------
    // Model hyperparameters
    // ----------------------------------------
    int input_dim = 2;
    int hidden_dim = 10;
    int output_dim = 2;

    float reg_lambda = 0.01f;
    float lr0 = 0.01f;
    float decay_k = 0.001f;

    int batch_size = 50;
    int num_passes = 2000;

    LRSchedule schedule = NULL;  // No LR schedule

    // ----------------------------------------
    // Dataset partitioning
    // ----------------------------------------
    int num_examples = count_lines(file_y);

    int block = num_examples / world_size;
    int start = world_rank * block;
    int end = (world_rank == world_size - 1) ? num_examples : start + block;
    int local_N = end - start;

    if (world_rank == 0)
    {
        printf("=============================================\n");
        printf(" MPI world size = %d\n", world_size);
        printf(" OpenMP threads = %d\n", omp_get_max_threads());
        printf(" Samples global = %d\n", num_examples);
        printf(" Samples per rank ≈ %d\n", block);
        printf("=============================================\n\n");
    }

    // ----------------------------------------
    // Load dataset fully (each rank loads all)
    // ----------------------------------------
    float *X = malloc(num_examples * input_dim * sizeof(float));
    int   *y = malloc(num_examples * sizeof(int));

    if (!X || !y)
    {
        fprintf(stderr, "[Rank %d] Memory allocation failed\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    load_X(file_X, X, num_examples, input_dim);
    load_y(file_y, y, num_examples);

    // ----------------------------------------
    // Initialize model
    // ----------------------------------------
    MLP *model = create_model(input_dim, hidden_dim, output_dim, ACT_TANH);

    // ----------------------------------------
    // Training
    // ----------------------------------------
    double t0 = MPI_Wtime();

    train(model,
          X + start * input_dim,
          y + start,
          local_N,
          lr0, reg_lambda,
          batch_size,
          num_passes,
          (world_rank == 0), // print loss only on rank 0
          schedule, decay_k);

    double t1 = MPI_Wtime();

    if (world_rank == 0)
        printf("\nTraining finished in %.3f seconds (MPI + OpenMP)\n", t1 - t0);

    // ----------------------------------------
    // Save weights (only rank 0)
    // ----------------------------------------
    if (world_rank == 0)
    {
        FILE *fw1 = fopen("output/W1.txt", "w");
        FILE *fb1 = fopen("output/b1.txt", "w");
        FILE *fw2 = fopen("output/W2.txt", "w");
        FILE *fb2 = fopen("output/b2.txt", "w");

        if (!fw1 || !fb1 || !fw2 || !fb2)
        {
            fprintf(stderr, "Could not open output files\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        for (int i = 0; i < input_dim * hidden_dim; i++)
            fprintf(fw1, "%f\n", model->W1[i]);

        for (int i = 0; i < hidden_dim; i++)
            fprintf(fb1, "%f\n", model->b1[i]);

        for (int i = 0; i < hidden_dim * output_dim; i++)
            fprintf(fw2, "%f\n", model->W2[i]);

        for (int i = 0; i < output_dim; i++)
            fprintf(fb2, "%f\n", model->b2[i]);

        fclose(fw1);
        fclose(fb1);
        fclose(fw2);
        fclose(fb2);

        printf("Weights saved in output/[W1 b1 W2 b2].txt\n");
    }

    // ----------------------------------------
    // Cleanup
    // ----------------------------------------
    free_model(model);
    free(X);
    free(y);

    MPI_Finalize();
    return 0;
}
