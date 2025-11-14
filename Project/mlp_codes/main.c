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
    int input_dim  = 2;
    int hidden_dim = 10;
    int output_dim = 2;

    float reg_lambda = 0.01f;
    float lr0        = 0.01f;
    float decay_k    = 0.001f;

    int batch_size = 50;
    int num_passes = 2000;

    LRSchedule schedule = NULL;  // no scheduling

    // ----------------------------------------
    // Split dataset for MPI processes
    // ----------------------------------------
    int num_examples = count_lines(file_y);

    int block = num_examples / world_size;
    int start = world_rank * block;
    int end   = (world_rank == world_size - 1) ? num_examples : start + block;
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
    // Load full dataset (every rank loads fully)
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
    // Create and initialize MLP
    // ----------------------------------------
    MLP *model = create_model(input_dim, hidden_dim, output_dim, ACT_TANH);

    // ----------------------------------------
    // Train
    // ----------------------------------------
    double t0 = MPI_Wtime();

    train(model,
          X + start * input_dim,
          y + start,
          local_N,
          lr0, reg_lambda,
          batch_size,
          num_passes,
          1,               // print loss on rank 0
          schedule, decay_k);

    double t1 = MPI_Wtime();

    if (world_rank == 0)
        printf("\nTraining finished in %.3f seconds (MPI + OpenMP)\n", t1 - t0);

    // ----------------------------------------
    // Cleanup
    // ----------------------------------------
    free_model(model);
    free(X);
    free(y);

    MPI_Finalize();
    return 0;
}
