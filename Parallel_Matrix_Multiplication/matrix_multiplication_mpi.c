#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 5  // matrix size (NxN), can be any integer

// Helper function to print a matrix stored in row-major order
void print_matrix(const char *name, double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", M[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Serial matrix multiplication: C = A * B
void serial_matmul(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i*n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Declare matrices
    double *A = NULL;
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));
    double *C_serial = malloc(N * N * sizeof(double));

    if (!B || !C || !C_serial) {
        fprintf(stderr, "Memory allocation failed!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize matrices on rank 0
    if (rank == 0) {
        A = malloc(N * N * sizeof(double));
        if (!A) {
            fprintf(stderr, "Memory allocation failed for A!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i*N + j] = i + j;
                B[i*N + j] = i - j;
            }
        }

        // Print input matrices
        print_matrix("Matrix A", A, N, N);
        print_matrix("Matrix B", B, N, N);

        // Serial multiplication and timing
        double serial_start = MPI_Wtime();
        serial_matmul(A, B, C_serial, N);
        double serial_end = MPI_Wtime();
        printf("Serial matrix C:\n");
        print_matrix("C_serial", C_serial, N, N);
        printf("Serial execution time: %f seconds\n", serial_end - serial_start);
    }

    // Prepare Scatterv parameters for rows (handles uneven division)
    int rows_per_proc = N / size;
    int rem = N % size;

    int sendcounts[size];
    int displs[size];
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int rows = (i < rem) ? rows_per_proc + 1 : rows_per_proc;
        sendcounts[i] = rows * N;      // number of elements
        displs[i] = offset;
        offset += sendcounts[i];
    }

    // Determine local number of rows for this rank
    int local_rows = (rank < rem) ? rows_per_proc + 1 : rows_per_proc;
    double *local_A = malloc(local_rows * N * sizeof(double));
    double *local_C = malloc(local_rows * N * sizeof(double));
    if (!local_A || !local_C) {
        fprintf(stderr, "Memory allocation failed for local matrices!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Scatter rows of A to all processes
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 local_A, local_rows*N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Broadcast B to all processes
    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Parallel multiplication
    MPI_Barrier(MPI_COMM_WORLD);
    double parallel_start = MPI_Wtime();

    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i*N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                local_C[i*N + j] += local_A[i*N + k] * B[k*N + j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double parallel_end = MPI_Wtime();

    // Gather results from all processes
    MPI_Gatherv(local_C, local_rows*N, MPI_DOUBLE,
                C, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Rank 0 prints result and computes speedup
    if (rank == 0) {
        printf("Parallel matrix C:\n");
        print_matrix("C_parallel", C, N, N);

        double parallel_time = parallel_end - parallel_start;
        printf("Time taken for parallel matrix multiplication: %f seconds\n", parallel_time);

        // Verify correctness
        int correct = 1;
        for (int i = 0; i < N*N; i++) {
            if (fabs(C[i] - C_serial[i]) > 1e-6) {
                correct = 0;
                break;
            }
        }
        if (correct)
            printf("Result verification: PASSED\n");
        else
            printf("Result verification: FAILED\n");

        // Free matrix A
        free(A);
    }

    // Free local matrices
    free(local_A);
    free(local_C);
    free(B);
    free(C);
    free(C_serial);

    MPI_Finalize();
    return 0;
}
