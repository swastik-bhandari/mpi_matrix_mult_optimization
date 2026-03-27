cmake_minimum_required(VERSION 3.14)
project(mpi_matmul CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(naive_matmul src/naive_matmul.cpp)
target_link_libraries(naive_matmul MPI::MPI_CXX)
target_compile_options(naive_matmul PRIVATE -O3 -march=native)

add_executable(cannon_matmul src/cannon_matmul.cpp)
target_link_libraries(cannon_matmul MPI::MPI_CXX)
target_compile_options(cannon_matmul PRIVATE -O3 -march=native)