#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 500 // Matrix size (NxN)

void print_matrix(const char *name, double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", M[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

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

double start_time, end_time;
double serial_start, serial_end;

int main(int argc, char *argv[]) {
	int rank, size;
	int rows_per_proc;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	static double A[N][N]; 
	static double B[N][N]; 
	static double C[N][N];
	static double local_A[N][N]; 
	static double local_C[N][N];

	static double C_serial[N][N];

	rows_per_proc=N/size;

	if (rank == 0) {
        // Initialize matrices
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i][j] = i + j;
                B[i][j] = i - j;
            }
    }

// 	if (rank == 0) {
// 		print_matrix("Matrix A", &A[0][0], N, N);
// print_matrix("Matrix B", &B[0][0], N, N);
// 	}


if (rank == 0) {
    serial_start = MPI_Wtime();
    serial_matmul(&A[0][0], &B[0][0], &C_serial[0][0], N);
    serial_end = MPI_Wtime();

    printf("Serial execution time: %f seconds\n",
           serial_end - serial_start);
}
	// Scatter rows of A
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,
                local_A, rows_per_proc * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

	// Broadcast B
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			

	start_time = MPI_Wtime();
	// Compute local matrix multiplication
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i][j] = 0.0;
            for (int k = 0; k < N; k++)
                local_C[i][j] += local_A[i][k] * B[k][j];
        }
    }

	end_time = MPI_Wtime();
	if (rank == 0) {
    printf("Time taken for parallel matrix multiplication: %f seconds\n",
           end_time - start_time);
}

	// Gather results
    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE,
               C, rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

		   
	// if (rank == 0) {
    //     printf("Matrix multiplication completed using %d processes.\n", size);
    // }


	MPI_Finalize();
	return 0;
}
