# MLP MPI + OpenMP Project

This project implements a **Multilayer Perceptron (MLP)** with a single hidden layer, trained on datasets like the 2D moons dataset. It supports **parallel training using MPI** (data parallelism) and **OpenMP** (multi-threaded matrix operations). It also supports dynamic learning rate schedules.

---

## Features

- Single hidden layer MLP
- Activation functions: Tanh, ReLU, Sigmoid, Leaky ReLU
- Cross-entropy loss with softmax output
- L2 regularization
- Mini-batch gradient descent
- Dynamic learning rate schedules:
  - Inverse time decay
  - Exponential decay
- Parallelization:
  - **MPI**: distribute batches across processes
  - **OpenMP**: multi-threaded matrix operations
- Thread-safe random initialization (Box-Muller)

---

## Requirements

- C compiler with OpenMP support (e.g., `gcc`, `mpicc`)
- MPI library (e.g., `OpenMPI`, `MPICH`)
- Make

Tested on Linux environments.

---

## Compilation

```bash
make
This will produce the executable:

nginx
Copy code
mlp
The Makefile includes optimization flags (-O3 -march=native -ffast-math) and OpenMP support.

Dataset
Example dataset included: data/data_X.txt and data/data_y.txt (10000 samples).

data_X.txt → input features (one row per sample)

data_y.txt → labels (integer class indices)

You can generate your own dataset using Python or any other tool.

Usage
Run with MPI and a single OpenMP thread
bash
Copy code
export OMP_NUM_THREADS=1
mpirun -np 4 ./mlp
-np 4 → number of MPI processes

OMP_NUM_THREADS=1 → number of threads per process

Run with OpenMP multi-threading (no MPI)
bash
Copy code
export OMP_NUM_THREADS=8
mpirun -np 1 ./mlp
1 MPI process using 8 OpenMP threads

Adjusting hyperparameters
In main.c:

c
Copy code
int input_dim = 2;       // number of input features
int hidden_dim = 128;    // number of hidden units
int output_dim = 2;      // number of output classes
float reg_lambda = 0.01f;
float lr0 = 0.01f;
int batch_size = 100;    // mini-batch size
int num_passes = 2000;   // number of epochs
float decay_k = 0.001f;  // learning rate decay parameter
Tips:

Increasing hidden_dim increases model capacity but also memory and computation.

batch_size can be tuned according to hidden_dim and number of threads. Larger batches may benefit multi-threading.

Use OMP_NUM_THREADS to utilize multiple cores for matrix operations.

MPI: Increasing processes distributes data across nodes but reduces local batch size. Adjust batch size accordingly.

Output
The program prints:

MPI world size and OpenMP threads

Samples per rank

Epoch loss (global or summed across MPI ranks)

Training time

Example:

java
Copy code
MPI world size = 4
OpenMP threads = 2
Samples global = 10000
Samples per rank ≈ 2500

Epoch 0 / 2000  Loss (sum over ranks): 2.45
...
Training finished in 1.82 seconds (MPI + OpenMP)
Code Structure
main.c → program entry, dataset loading, training

model.c / model.h → MLP definition, forward/backward pass, training, gradient updates

utils.c / utils.h → matrix operations, activations, random number generators, learning rate schedules

data/ → example datasets
