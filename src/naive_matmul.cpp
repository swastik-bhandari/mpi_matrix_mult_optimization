#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

static void fill_matrix(std::vector<double>& M, int n) {
    for (int i = 0; i < n * n; i++)
        M[i] = static_cast<double>(rand()) / RAND_MAX;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = (argc > 1) ? atoi(argv[1]) : 512;

    if (n % size != 0) {
        if (rank == 0)
            std::cerr << "Matrix size must be divisible by number of processes\n";
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = n / size;

    std::vector<double> A, B(n * n), C;

    if (rank == 0) {
        srand(42);
        A.resize(n * n);
        C.resize(n * n, 0.0);
        fill_matrix(A, n);
        fill_matrix(B, n);
    }

    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> local_A(rows_per_proc * n);
    MPI_Scatter(A.data(), rows_per_proc * n, MPI_DOUBLE,
                local_A.data(), rows_per_proc * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    std::vector<double> local_C(rows_per_proc * n, 0.0);

    double t_start = MPI_Wtime();

    for (int i = 0; i < rows_per_proc; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = local_A[i * n + k];
            for (int j = 0; j < n; j++) {
                local_C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Gather(local_C.data(), rows_per_proc * n, MPI_DOUBLE,
               C.data(), rows_per_proc * n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "naive " << n << " " << size << " " << max_time << "\n";
    }

    MPI_Finalize();
    return 0;
}
