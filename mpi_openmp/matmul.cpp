#include<iostream>
#include<mpi.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include <ctime>

#include <omp.h>

#include "common.hpp"

#define logfileline printf("------file:%s, line:%d\n", __FILE__, __LINE__);

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

int splitA(int argc, char** argv)
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
        //matmul_cpu(A, B, Ccompare, m, n, p);
        cpu_end = clock();
        printf("%s time: %f\n", "CPU not optimized", (double)(cpu_end - cpu_start)/CLOCKS_PER_SEC);
        std::cout << "CPU not optimized result:" << std::endl;
        print_array(Ccompare, 10);
        print_array(Ccompare + m * n / 2 - 5, 10);
        print_array(Ccompare + m * n - 10, 10);

        cpu_start = clock();
        //matmul_cpu_openmp(A, B, C, m, n, p, omp_get_num_procs());
        cpu_end = clock();
        printf("%s time: %f\n", "CPU openmp", (double)(cpu_end - cpu_start)/CLOCKS_PER_SEC);
        std::cout << "CPU openmp result:" << std::endl;
        print_array(C, 10);
        print_array(C + m * n / 2 - 5, 10);
        print_array(C + m * n - 10, 10);
        std::cout << "compare CPU openMP result with not optimized:" << std::endl;
        //compare_arrays(Ccompare, C, m * n);
        memset(C, 0, m * n * sizeof(float));

        cpu_start = clock();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatter(A, bm * p, MPI_FLOAT, bA, bm * p, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, p * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    matmul_cpu(bA, B, bC, bm, n, p);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(bC, bm * n, MPI_FLOAT, C, bm * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int remainRowsStartId = bm * numprocs;
    if(myrank == 0 && remainRowsStartId < m) {
        int remainRows = m - remainRowsStartId;
        matmul_cpu(A + remainRowsStartId * p, B, C + remainRowsStartId * n, remainRows, n, p);
    }
    if(myrank == 0) {
        cpu_end = clock();
        printf("%s time: %f\n", "CPU MPI", (double)(cpu_end - cpu_start)/CLOCKS_PER_SEC);
        std::cout << "MPI result:" << std::endl;
        print_array(C, 10);
        print_array(C + m * n / 2 - 5, 10);
        print_array(C + m * n - 10, 10);
        std::cout << "compare MPI result with single thread:" << std::endl;
        //compare_arrays(Ccompare, C, m * n);
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

int splitAB(int argc, char** argv)
{
    int m = atoi(argv[1]);
    int p = atoi(argv[2]);
    int n = atoi(argv[3]);

    float *A, *B, *C, *Ccompare;
    float *bA, *bB, *bC;

    int myrank, numprocs;

    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int saveM=m; // 为了使行数据平均分配每一个进程
    if(m%numprocs!=0){
        m-=m%numprocs;
        m+=numprocs;
    }
    int saveN=n; // 为了使行数据平均分配每一个进程
    if(n%numprocs!=0){
        n-=n%numprocs;
        n+=numprocs;
    }

    int bm = m / numprocs;
    int bn = n / numprocs;

    bA = new float[p * bm];
    bB = new float[bn * p];
    bC = new float[n * bm];

    clock_t cpu_start, cpu_end;

    if(myrank == 0) {
        A = new float[m * p];
        B = new float[p * n];
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

        //logfileline
        cpu_start = clock();
        //0进程存储第一块B按列划分数据B0
        for(int i=0; i<p; i++)
            for(int j=0; j<bn; j++)
                bB[bn * i + j] = B[ n * i + j];
        //logfileline
        //发送B0到B(numprocs - 1）列数据
        for(int k=1; k<numprocs; k++) {
            int beginCol = bn * k;
            for(int i=0; i<p; i++)
                MPI_Send(B + n * i + beginCol, bn, MPI_FLOAT, k, i, MPI_COMM_WORLD);
        }
        //logfileline
        delete[] B;
    }
    // 分发A0到A(num_procs-1）行数据；
    MPI_Scatter(A, p * bm, MPI_FLOAT, bA, p * bm, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //logfileline
    if(myrank == 0) {
        delete[] A;
    }
    //logfileline

    // 接受B列数据
    if(myrank != 0) {
        for(int i=0; i<p; i++)
            MPI_Recv(bB + bn * i, bn, MPI_FLOAT, 0, i, MPI_COMM_WORLD, &status);
    }
    //logfileline

    //MPI_Barrier(MPI_COMM_WORLD);

    for(int k=0; k < numprocs; k++) {
        int beginCol = ((k + myrank) % numprocs) * bn;
        for(int i=0; i<bm; i ++) {
            for(int j=0; j<bn; j ++) {
                float temp = 0;
                for(int l=0; l<p; l++) {
                    temp += bA[p * i + l] * bB[bn * l + j];
                }
                bC[n*i + beginCol + j] = temp;
            }
        }
        //logfileline
        int dest = (myrank - 1 + numprocs) % numprocs;
        int src = (myrank + 1) % numprocs;
        /*
        int times = 0;
        do{
            times ++;
            printf("---before API, myrank:%d, k:%d, dest:%d, src:%d, time:%d\n", myrank, k, dest, src, times);
            MPI_Sendrecv_replace(bB, bn * p, MPI_FLOAT, dest, k, src, k, MPI_COMM_WORLD, &status);
            printf("---after  API, myrank:%d, k:%d, dest:%d, src:%d, status.MPI_ERROR:%d, time:%d\n", myrank, k, dest, src, status.MPI_ERROR, times);
        } while(status.MPI_ERROR != 0);
        */
        //MPI_Barrier(MPI_COMM_WORLD);
        //printf("---before API, myrank:%d, k:%d, dest:%d, src:%d\n", myrank, k, dest, src);
        MPI_Sendrecv_replace(bB, bn * p, MPI_FLOAT, dest, k, src, k, MPI_COMM_WORLD, &status);
        //printf("---after  API, myrank:%d, k:%d, dest:%d, src:%d, status.MPI_ERROR:%d\n", myrank, k, dest, src, status.MPI_ERROR);
    }

    //logfileline
    MPI_Gather(bC, bm * n, MPI_FLOAT, C, bm * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(myrank == 0) {
        cpu_end = clock();
        printf("%s time: %f\n", "CPU MPI", (double)(cpu_end - cpu_start)/CLOCKS_PER_SEC);

        std::cout << "MPI result:" << std::endl;
        print_array(C, 10);
        print_array(C + m * n / 2 - 5, 10);
        print_array(C + m * n - 10, 10);
        std::cout << "compare MPI result with single thread:" << std::endl;
        compare_arrays(Ccompare, C, m * n);
        delete[] C;
    }
    delete[] bA;
    delete[] bB;
    delete[] bC;

    MPI_Finalize();
    return 0;
}

int main(int argc, char** argv) {
    return splitA(argc, argv);
    //return splitAB(argc, argv);
}
/*
openmpi的MPI_Sendrecv_replace，会返回错误似乎没有阻塞，mpich没有这个问题
                                not optimized       openMP          MPI                        MPI(both A&B split)
3096 1024 2048 mpicc            13.85               28.40           11.27                      6.47
3096 1024 2048 gcc              13.88               28.47           10.28                      6.52
6192 1024 2048 gcc              32.66               55.39           20.96                      12.71
6192 2048 2048 gcc              172.31              334.50          82.16                      26.60
6192 2048 4096 gcc              531.84              957.95          160.71                     83.04

#split A
mpirun -np 3 /data0/nfsmnt/mpi/matmul 3096 1024 2048
#split A, multi hosts
mpirun -np 3 --host dtpct:1,mdubu:1,mdlapubu:1 /data0/nfsmnt/mpi/matmul 3096 1024 2048
#split A&B, multi hosts
mpirun -np 4 --host dtpct:2,mdubu:1,mdlapubu:1 /data0/nfsmnt/mpi/matmul 3096 1024 2048
*/
