#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4 // Matrix size (NxN)
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
	rows_per_proc=N/size;

	if (rank == 0) {
        // Initialize matrices
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i][j] = i + j;
                B[i][j] = i - j;
            }
    }

	// Scatter rows of A
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,
                local_A, rows_per_proc * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

	// Broadcast B
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			

	// Compute local matrix multiplication
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i][j] = 0.0;
            for (int k = 0; k < N; k++)
                local_C[i][j] += local_A[i][k] * B[k][j];
        }
    }

	// Gather results
    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE,
               C, rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

	if (rank == 0) {
        printf("Matrix multiplication completed using %d processes.\n", size);
    }


	MPI_Finalize();
	return 0;
}
