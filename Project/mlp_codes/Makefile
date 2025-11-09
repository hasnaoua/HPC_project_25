# -------------------------
# Compiler settings
# -------------------------
CC = gcc
CFLAGS = -O3 -march=native -ffast-math -Wall -g
TARGET = mlp

# Object files
OBJ = main.o model.o utils.o

# -------------------------
# Default target
# -------------------------
all: $(TARGET)

# -------------------------
# Link the executable
# -------------------------
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -lm -o $(TARGET)

# -------------------------
# Compile source files
# -------------------------
main.o: main.c model.h utils.h
	$(CC) $(CFLAGS) -c main.c

model.o: model.c model.h utils.h
	$(CC) $(CFLAGS) -c model.c

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c

# -------------------------
# Clean build files
# -------------------------
clean:
	rm -f *.o $(TARGET)
