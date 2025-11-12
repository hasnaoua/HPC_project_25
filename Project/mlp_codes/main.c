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

    const char *file_X = "data/data_X.txt";
    const char *file_y = "data/data_y.txt";

    int input_dim = 2, output_dim = 2, hidden_dim = 10;
    float reg_lambda = 0.01f, lr = 0.01f;

    int num_examples = count_lines(file_y);
    int start = world_rank * (num_examples / world_size);
    int end   = (world_rank == world_size - 1)
              ? num_examples
              : (start + num_examples / world_size);
    int local_N = end - start;

    if (world_rank == 0)
        printf("MPI world size = %d | Using OpenMP with up to %d threads\n",
               world_size, omp_get_max_threads());

    float *X = malloc(num_examples * input_dim * sizeof(float));
    int   *y = malloc(num_examples * sizeof(int));
    load_X(file_X, X, num_examples, input_dim);
    load_y(file_y, y, num_examples);

    MLP *model = create_model(input_dim, hidden_dim, output_dim, ACT_TANH);

    double t0 = MPI_Wtime();
    train(model, X + start * input_dim, y + start, local_N, lr, reg_lambda, 50, 2000, 1);
    double t1 = MPI_Wtime();

    if (world_rank == 0)
        printf("Training finished in %.3f sec (MPI+OpenMP)\n", t1 - t0);

    free_model(model);
    free(X);
    free(y);

    MPI_Finalize();
    return 0;
}
