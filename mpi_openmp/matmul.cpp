#include<iostream>
#include<mpi.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include <ctime>

#include <omp.h>

#include "common.hpp"

template <class T>
void matmul_cpu(T * mat_a, T * mat_b, T * mat_out, int m, int n, int k)
{
    for(int x = 0; x < m; x ++)
    {
        for(int y = 0; y < n; y ++)
        {
            float temp = 0;
            for(int x2y = 0; x2y < k; x2y ++)
            {
                temp += *(mat_a + k * x + x2y) * (*(mat_b + n * x2y + y));
            }
            *(mat_out + n * x + y) = temp;
        }
    }
}

template <class T>
void matmul_cpu_openmp(T * mat_a, T * mat_b, T * mat_out, int m, int n, int k, int num_procs)
{
    #pragma omp parallel for num_threads(num_procs)
    for(int x = 0; x < m; x ++)
    {
        //printf("thread_num:%d\n", omp_get_thread_num());
        for(int y = 0; y < n; y ++)
        {
            float temp = 0;
            for(int x2y = 0; x2y < k; x2y ++)
            {
                temp += *(mat_a + k * x + x2y) * (*(mat_b + n * x2y + y));
            }
            *(mat_out + n * x + y) = temp;
        }
    }
}

template <class T>
void tranpose_matrix(T * input, T * output, int m, int n)
{
    for(int x = 0; x < m; x ++)
    {
        for(int y = 0; y < n; y ++)
        {
            *(output + m * y + x) = *(input + n * x + y);
        }
    }
}

int main(int argc, char** argv)
{
    int m = atoi(argv[1]);
    int p = atoi(argv[2]);
    int n = atoi(argv[3]);

    float *A, *B, *C, *Ccompare;
    float *bA, *bC;

    int myrank, numprocs;

    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int bm = m / numprocs;

    bA = new float[bm * p];
    B = new float[p * n];
    bC = new float[bm * n];

    clock_t cpu_start, cpu_end;

    if(myrank == 0) {
        A = new float[m * p];
        C = new float[m * n];

        initialize(A, m*p, INIT_RANDOM);
        initialize(B, p*n, INIT_RANDOM);

        Ccompare = new float[m * n];
        cpu_start = clock();
        matmul_cpu(A, B, Ccompare, m, n, p);
        cpu_end = clock();
        printf("%s time: %f\n", "CPU not optimized", (double)(cpu_end - cpu_start)/CLOCKS_PER_SEC);
        std::cout << "CPU not optimized result:" << std::endl;
        print_array(Ccompare, 10);
        print_array(Ccompare + m * n / 2 - 5, 10);
        print_array(Ccompare + m * n - 10, 10);

        cpu_start = clock();
        matmul_cpu_openmp(A, B, C, m, n, p, omp_get_num_procs());
        cpu_end = clock();
        printf("%s time: %f\n", "CPU openmp", (double)(cpu_end - cpu_start)/CLOCKS_PER_SEC);
        std::cout << "CPU openmp result:" << std::endl;
        print_array(C, 10);
        print_array(C + m * n / 2 - 5, 10);
        print_array(C + m * n - 10, 10);
        std::cout << "compare CPU openMP result with not optimized:" << std::endl;
        compare_arrays(Ccompare, C, m * n);
        memset(C, 0, m * n * sizeof(float));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatter(A, bm * p, MPI_FLOAT, bA, bm * p, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, p * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    cpu_start = clock();
    matmul_cpu(bA, B, bC, bm, n, p);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(bC, bm * n, MPI_FLOAT, C, bm * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int remainRowsStartId = bm * numprocs;
    if(myrank == 0 && remainRowsStartId < m) {
        int remainRows = m - remainRowsStartId;
        matmul_cpu(A + remainRowsStartId * p, B, C + remainRowsStartId * n, remainRows, n, p);
    }
    cpu_end = clock();
    printf("%s time: %f\n", "CPU MPI", (double)(cpu_end - cpu_start)/CLOCKS_PER_SEC);
    if(myrank == 0) {
        std::cout << "MPI result:" << std::endl;
        print_array(C, 10);
        print_array(C + m * n / 2 - 5, 10);
        print_array(C + m * n - 10, 10);
        std::cout << "compare MPI result with single thread:" << std::endl;
        compare_arrays(Ccompare, C, m * n);
    }
    delete[] bA;
    delete[] B;
    delete[] bC;

    if(myrank == 0) {
        delete[] A;
        delete[] C;
    }

    MPI_Finalize();
    return 0;
}

/*
                                not optimized       openMP          MPI                        MPI(B scattered)
3096 1024 2048                  13.85               28.40           14.75/14.82/14.85          14.75/14.82/14.85
3096 1024 2048 gcc compiler     13.88               28.47           11.83/11.88/11.91          14.75/14.82/14.85
6192 1024 2048                  27.80               56.70           22.00/22.13/22.15
6192 2048 2048                  139.04              274.57          108.53/108.67/108.72

#split A
mpirun -np 3 /data0/nfsmnt/mpi/matmul 3096 1024 2048
#split A, multi hosts
mpirun -np 3 --host dtpct:1,mdubu:1,mdlapubu:1 /data0/nfsmnt/mpi/matmul 3096 1024 2048
#split A&B, multi hosts
mpirun -np 4 --host dtpct:2,mdubu:1,mdlapubu:1 /data0/nfsmnt/mpi/matmul 3096 1024 2048
*/
