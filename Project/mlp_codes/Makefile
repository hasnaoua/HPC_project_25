# =========================================================
# Compiler settings
# =========================================================
# Use MPI compiler wrapper for hybrid parallelism
CC = mpicc

# Optimization and parallel flags
CFLAGS = -O3 -march=native -ffast-math -Wall -g -fopenmp
LDFLAGS = -fopenmp -lm

# Target name
TARGET = mlp

# Object files
OBJ = main.o model.o utils.o

# =========================================================
# Default target
# =========================================================
all: $(TARGET)

# =========================================================
# Link the executable
# =========================================================
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $(TARGET)

# =========================================================
# Compile source files
# =========================================================
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# =========================================================
# Run targets
# =========================================================
run:
	./$(TARGET)

# Run with OpenMP threads
run_omp:
	OMP_NUM_THREADS=2 ./$(TARGET)

# Run with MPI processes (and OpenMP inside each process)
run_mpi:
	mpirun -np 1 ./$(TARGET)

# =========================================================
# Clean build files
# =========================================================
clean:
	rm -f *.o $(TARGET)
