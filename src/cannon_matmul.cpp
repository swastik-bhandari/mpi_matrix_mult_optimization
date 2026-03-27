#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cassert>

static int grid_rank(int row, int col, int q) {
    return row * q + col;
}

static void local_matmul(const std::vector<double>& A,
                          const std::vector<double>& B,
                          std::vector<double>& C,
                          int block) {
    for (int i = 0; i < block; i++) {
        for (int k = 0; k < block; k++) {
            double a_ik = A[i * block + k];
            for (int j = 0; j < block; j++) {
                C[i * block + j] += a_ik * B[k * block + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int q = static_cast<int>(round(sqrt(static_cast<double>(size))));
    if (q * q != size) {
        if (rank == 0)
            std::cerr << "Cannon's algorithm requires a perfect-square number of processes\n";
        MPI_Finalize();
        return 1;
    }

    int n = (argc > 1) ? atoi(argv[1]) : 512;
    if (n % q != 0) {
        if (rank == 0)
            std::cerr << "n must be divisible by sqrt(p)\n";
        MPI_Finalize();
        return 1;
    }

    int block = n / q;
    int my_row = rank / q;
    int my_col = rank % q;

    int dims[2] = {q, q};
    int periods[2] = {1, 1};
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    my_row = coords[0];
    my_col = coords[1];

    std::vector<double> full_A, full_B;
    std::vector<double> local_A(block * block), local_B(block * block), local_C(block * block, 0.0);

    if (rank == 0) {
        srand(42);
        full_A.resize(n * n);
        full_B.resize(n * n);
        for (int i = 0; i < n * n; i++) {
            full_A[i] = static_cast<double>(rand()) / RAND_MAX;
            full_B[i] = static_cast<double>(rand()) / RAND_MAX;
        }
    }

    std::vector<double> send_A(block * block), send_B(block * block);
    std::vector<MPI_Request> requests;

    if (rank == 0) {
        for (int pr = 0; pr < q; pr++) {
            for (int pc = 0; pc < q; pc++) {
                int dest = grid_rank(pr, pc, q);
                std::vector<double> tmp_A(block * block), tmp_B(block * block);
                for (int i = 0; i < block; i++) {
                    for (int j = 0; j < block; j++) {
                        tmp_A[i * block + j] = full_A[(pr * block + i) * n + pc * block + j];
                        tmp_B[i * block + j] = full_B[(pr * block + i) * n + pc * block + j];
                    }
                }
                if (dest == 0) {
                    local_A = tmp_A;
                    local_B = tmp_B;
                } else {
                    MPI_Send(tmp_A.data(), block * block, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
                    MPI_Send(tmp_B.data(), block * block, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
                }
            }
        }
    } else {
        MPI_Recv(local_A.data(), block * block, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_B.data(), block * block, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    {
        int shift_src, shift_dst;
        MPI_Cart_shift(grid_comm, 1, -my_row, &shift_src, &shift_dst);
        MPI_Sendrecv_replace(local_A.data(), block * block, MPI_DOUBLE,
                             shift_dst, 10, shift_src, 10, grid_comm, MPI_STATUS_IGNORE);

        MPI_Cart_shift(grid_comm, 0, -my_col, &shift_src, &shift_dst);
        MPI_Sendrecv_replace(local_B.data(), block * block, MPI_DOUBLE,
                             shift_dst, 11, shift_src, 11, grid_comm, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(grid_comm);
    double t_start = MPI_Wtime();

    int left, right, up, down;
    MPI_Cart_shift(grid_comm, 1, -1, &right, &left);
    MPI_Cart_shift(grid_comm, 0, -1, &down, &up);

    std::vector<double> tmp_A(block * block), tmp_B(block * block);

    for (int step = 0; step < q; step++) {
        local_matmul(local_A, local_B, local_C, block);

        MPI_Sendrecv(local_A.data(), block * block, MPI_DOUBLE, left, 20,
                     tmp_A.data(), block * block, MPI_DOUBLE, right, 20,
                     grid_comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(local_B.data(), block * block, MPI_DOUBLE, up, 21,
                     tmp_B.data(), block * block, MPI_DOUBLE, down, 21,
                     grid_comm, MPI_STATUS_IGNORE);

        local_A = tmp_A;
        local_B = tmp_B;
    }

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "cannon " << n << " " << size << " " << max_time << "\n";
    }

    MPI_Comm_free(&grid_comm);
    MPI_Finalize();
    return 0;
}
