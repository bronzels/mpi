//
// Created by Apple on 2022/11/3.
//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//

void omp_thread_test()
{
    int coreNum = omp_get_num_procs();
    printf("coreNum = %d\n", coreNum); //设置线程数，一般设置的线程数不超过cpu核心数，这里开4个县城执行并行代码段
    omp_set_num_threads(coreNum); //
#pragma omp parallel for
    for(int i = 0; i < 4 * coreNum; i++)
        printf("i = %d, I am Thread %d\n", i, omp_get_thread_num());
}

//

double Test(int n)
{
    double t = 0.0;
    for ( int i = 0; i < 100000000; ++i) {
        t += n*n;
    }
    //printf("n = %d, %f I am Thread %d\n", n, t, omp_get_thread_num());
    return t;
}

void omp_for_test1()
{
    double t_avg = 0.0;
    clock_t t2 = clock();
    omp_set_num_threads(8);
#pragma omp parallel for reduction(+: t_avg)
    for (int i = 0; i < 10; ++i) {
        t_avg += Test(i);
    }
    t_avg = t_avg / 10;

    clock_t t3 = clock();
    printf("t_avg = %f, Parallel time: %ld\n", t_avg, t3 - t2);
}

void omp_for_test2()
{
    double t_avg = 0.0;
    clock_t t2 = clock();
    omp_set_num_threads(8);
#pragma omp parallel for
    for (int i = 0; i < 10; ++i) {
#pragma omp critical
        t_avg += Test(i);
    }
    t_avg = t_avg / 10;

    clock_t t3 = clock();
    printf("t_avg = %f, Parallel time: %d\n", t_avg, t3 - t2);
}

//
void no_opm_pi()
{
    double factor = 1;
    double sum = 0.0;
    int n = 100000000;
    clock_t t2 = clock();
    for (int i = 0; i < n; i++)
    {
        sum += factor / (2 * i + 1);
        factor = -factor;
    }
    double pi_approx = 4.0*sum;
    clock_t t3 = clock();
    printf("%f  %ld\n", pi_approx, t3 - t2);
}

void opm_for_pi()
{
    double factor = 1;
    double sum = 0.0;
    int n = 1000000000;
    clock_t t2 = clock();
#pragma omp parallel for num_threads(omp_get_num_procs()) reduction(+:sum) private(factor)
    for (int i = 0; i < n; i++)
    {
        if ( i % 2 == 0)
            factor = 1.0;
        else
            factor = -1.0;
        sum += factor / (2 * i + 1);
    }
    double pi_approx = 4.0*sum;
    clock_t t3 = clock();
    printf("%f  %ld\n", pi_approx, t3 - t2);
}
//
//
//

static long num_steps = 1000000000;//越大值越精确
const int NUM_THREADS = 8;
void no_omp_pi2()
{
    int i;
    double x = 0.0, pi = 0.0, sum = 0.0;
    double step = 1.0 / (double)num_steps;
    clock_t t2 = clock();
    for (i = 1; i <= num_steps; i++)
    {
        x = (i - 0.5)*step;
        sum = sum + 4.0 / (1.0 + x*x);
    }
    pi = step*sum;
    clock_t t3 = clock();
    printf("no_omp_pi2 : %1.12f %d\n", pi, t3 - t2);
}

void omp_for_pi2_1()
{
    int i;
    double pi = 0.0;
    double sum[NUM_THREADS];
    double step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);
    clock_t t2 = clock();
#pragma omp parallel private(i)  //并行域开始，每个线程(0和1)都会执行该代码
    {
        double x;
        int id = omp_get_thread_num();
        for (i = id, sum[id] = 0.0; i < num_steps; i = i + NUM_THREADS)
        {
            x = (i + 0.5)*step;
            sum[id] += 4.0 / (1.0 + x*x);
        }
    }
    for (i = 0; i < NUM_THREADS; i++)
        pi += sum[i] * step;
    clock_t t3 = clock();
    printf("omp_for_pi2_1: %1.12f %d\n", pi, t3 - t2);
}

void omp_for_pi2_2()
{
    int i;
    int id;
    double pi = 0.0, sum[NUM_THREADS] = { 0.0 };
    double step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);  //设置2线程
    clock_t t2 = clock();
#pragma omp parallel  //并行域开始，每个线程都会执行该代码
    {
        double x;

        id = omp_get_thread_num();
        sum[id] = 0;
#pragma omp for  //未指定chunk，迭代平均分配给各线程，连续划分
        for (i = 1; i <= num_steps; i++) {
            x = (i + 0.5)*step;
            sum[id] += 4.0 / (1.0 + x*x);
        }
    }
    for (i = 0; i < NUM_THREADS; i++)
        pi += sum[i] * step;
    clock_t t3 = clock();
    printf("omp_for_pi2_2: %1.12f %d\n", pi, t3 - t2);
}

void omp_for_pi2_3()
{
    int i;
    double pi = 0.0;
    double sum = 0.0;
    int id = 0;
    double x = 0.0;
    double step = 1.0 / (double)num_steps;
    clock_t t2 = clock();
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel private(x,sum,id) //该子句表示x,sum变量对于每个线程是私有的(!!!注意id要设为私有变量)
    {
        id = omp_get_thread_num();
        for (i = id, sum = 0.0; i <= num_steps; i = i + NUM_THREADS)
        {
            x = (i + 0.5)*step;
            sum += 4.0 / (1.0 + x*x);
        }
#pragma omp critical  //指定代码段在同一时刻只能由一个线程进行执行
        pi += sum*step;
    }
    clock_t t3 = clock();
    printf("omp_for_pi2_3: %1.12f %d\n", pi, t3 - t2);
}

void omp_for_pi2_4()
{
    int i;
    double pi = 0.0;
    double sum = 0.0;
    double x = 0.0;
    double step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);  //设置2线程
    clock_t t2 = clock();
#pragma omp parallel for reduction(+:sum) private(x) //每个线程保留一份私有拷贝sum，x为线程私有，最后对线程中所以sum进行+规约，并更新sum的全局值
    for (i = 1; i <= num_steps; i++) {
        x = (i - 0.5)*step;
        sum += 4.0 / (1.0 + x*x);
    }
    pi = sum * step;
    clock_t t3 = clock();
    printf("omp_for_pi2_4: %1.12f %d\n", pi, t3 - t2);
}

//
//
//
void omp_sleep()
{
    int coreNum = omp_get_num_procs();
    printf("coreNum = %d\n", coreNum);
    omp_set_num_threads(coreNum);	//设置线程数，一般设置的线程数不超过CPU核心数，这里开4个线程执行并行代码段
#pragma omp parallel for
    for (int i = 0; i < coreNum; i++)
    {
        while (1)
        {
            int a = 5;
            int b = a;
        }
    }
}
//
//
//
int main(int argc, char* argv[])
{
    //omp_thread_test();
    //omp_for_test1();
    //omp_for_test2();
    //no_opm_pi();
    //opm_for_pi();
    //no_omp_pi2();
    //omp_for_pi2_1();
    //omp_for_pi2_2();
    //omp_for_pi2_3();
    //omp_for_pi2_4();
    omp_sleep();
    system("pause");
}
