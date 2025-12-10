# MLP Project 
 

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
- Create in your local your own folder named data (for stroring features and labels)
- Create in your local your own folder named output (for stroring weights final updated values after training)

Tested on Linux environments.

---

## Compilation

# Build

This will produce the executable:

```
make
```

Output binary:

```
mlp
```

**Notes:**  
The Makefile uses optimization flags:

- `-O3`  
- `-march=native`  
- `-ffast-math`  
- OpenMP support enabled

---

# Dataset

Example dataset (10,000 samples):

- `data/data_X.txt` → input features (one row per sample)  
- `data/data_y.txt` → labels (integer class indices)

You can generate your own dataset with Python or any tool.

---

# Usage

## Run with MPI and a single OpenMP thread

**Command:**
```
export OMP_NUM_THREADS=1
mpirun -np 4 ./mlp
```

**Explanation:**
- `-np 4` → number of MPI processes  
- `OMP_NUM_THREADS=1` → one OpenMP thread per process

---

## Run with OpenMP multi-threading (no MPI)

**Command:**
```
export OMP_NUM_THREADS=8
mpirun -np 1 ./mlp
```

**Explanation:**  
1 MPI process using 8 OpenMP threads.

---

# Adjusting Hyperparameters

In `main.c`:

```c
int input_dim = 2;       // number of input features
int hidden_dim = 128;    // number of hidden units
int output_dim = 2;      // number of output classes
float reg_lambda = 0.01f;
float lr0 = 0.01f;
int batch_size = 100;    // mini-batch size
int num_passes = 2000;   // number of epochs
float decay_k = 0.001f;  // learning rate decay parameter
```

### Tips
- Increasing `hidden_dim` increases model capacity (and cost).
- `batch_size` can be tuned depending on `hidden_dim` and thread count.
- Use `OMP_NUM_THREADS` to exploit multi-core matrix ops.
- Under MPI: more ranks = smaller local batch → adjust `batch_size`.

---

# Output

The program prints:

- MPI world size  
- OpenMP thread count  
- Samples per rank  
- Epoch loss  
- Training time  

**Example:**

```
MPI world size = 4
OpenMP threads = 2
Samples global = 10000
Samples per rank ≈ 2500

Epoch 0 / 2000  Loss (sum over ranks): 2.45
...
Training finished in 1.82 seconds (MPI + OpenMP)
```

---

# Code Structure

- `main.c` → program entry, dataset loading, training loop  
- `model.c` / `model.h` → MLP, forward/backward pass, updates  
- `utils.c` / `utils.h` → matrix ops, activations, RNG, LR schedules  
- `data/` → example datasets
