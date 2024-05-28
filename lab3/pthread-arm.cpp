#include <iostream>
#include <sys/time.h>
#include <pthread.h>
#include <semaphore.h>
#include <arm_neon.h>
#include <cmath>
#include <string>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <random>
using namespace std;

//------------------------------------------ 线程控制变量 ------------------------------------------
typedef struct
{
    int k;
    int t_id;
} threadParam_t;

sem_t sem_Division;
pthread_barrier_t barrier;
pthread_mutex_t task;
int indexkey = 0;
const int threadnum =20;

sem_t sem_main;
sem_t sem_workerstart[threadnum]; // 每个线程有自己专属的信号量
sem_t sem_workerend[threadnum];

sem_t sem_leader;
sem_t sem_Divsion1[(threadnum)-1];
sem_t sem_Elimination[threadnum-1];

sem_t sem_leader2;
sem_t sem_Divsion2[(threadnum)-1];
sem_t sem_Elimination2[threadnum - 1];


pthread_barrier_t barrier_Elimination;
pthread_barrier_t barrier_Divsion;


//pthread_mutex_t task; // 用于任务分配的互斥锁
pthread_mutex_t division_done; // 用于控制除法完成的互斥锁

pthread_mutex_t task_mutex;
pthread_cond_t cond_division_done;



// ------------------------------------------ 全局计算变量 ------------------------------------------
const int N =1000;
const int L = 100;
const int LOOP =2;
float datanum[N][N];
float matrix[N][N];


// 初始化data，保证每次数据都是一致的
void init_data()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = i; j < N; j++)
        {
            datanum[i][j] = rand() * 1.0 / RAND_MAX * L;
        }
    }
    for (int i = 0; i < N - 1; i++)
        for (int j = i + 1; j < N; j++)
            for (int k = 0; k < N; k++)
                datanum[j][k] += datanum[i][k];
}

// 用data初始化matrix，保证每次进行计算的数据是一致的
void init_matrix()
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = datanum[i][j];
}

// 串行算法
void calculate_serial()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

//动态线程函数
void* threadFunc1(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    int t_id = p->t_id;
    int i = k + t_id + 1;
    for (int j = k + 1; j < N; ++j)
    {
        matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
    }
    matrix[i][k] = 0;
    pthread_exit(NULL);
    return NULL;
}

//动态线程版本
void serial()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        int worker_count =N - 1 - k;
        pthread_t* handles = (pthread_t*)malloc(worker_count * sizeof(pthread_t)); // 创建对应的 Handles
        threadParam_t* param = (threadParam_t*)malloc(worker_count * sizeof(threadParam_t)); // 创建对应的线程数据结构

        for (int t_id =0; t_id<worker_count; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_create(&handles[t_id], NULL, threadFunc1, (void*)&param[t_id]);
        }
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            pthread_join(handles[t_id], NULL);
        }
        free(handles);
        free(param);
    }

}

// pthread_dynamci 线程函数 互斥锁
void* threadFunc_mutex(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++) {
        if (t_id == 0) {
            pthread_mutex_lock(&task_mutex);
            for (int j = k + 1; j < N; j++) {
                matrix[k][j] /= matrix[k][k];
            }
            matrix[k][k] = 1.0;
            indexkey = k + 1;
            pthread_cond_broadcast(&cond_division_done);
            pthread_mutex_unlock(&task_mutex);
        }
        else {
            pthread_mutex_lock(&task_mutex);
            while (indexkey <= k) {
                pthread_cond_wait(&cond_division_done, &task_mutex);
            }
            pthread_mutex_unlock(&task_mutex);
        }

        for (int i = k + 1 + t_id; i < N; i += threadnum) {
            for (int j = k + 1; j < N; j++) {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0.0;
        }
    }
    pthread_exit(NULL);
    return NULL;
}

// pthread_dynamic 并行算法 动态划分 三重循环全进去  互斥锁
void calculate_mutex()
{
    pthread_mutex_init(&task_mutex, NULL);
    pthread_cond_init(&cond_division_done, NULL);

    pthread_t threads[threadnum];
    threadParam_t params[threadnum];
    for (int i = 0; i < threadnum; i++) {
        params[i].t_id = i;
        pthread_create(&threads[i], NULL, threadFunc_mutex, &params[i]);
    }

    for (int i = 0; i < threadnum; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&task_mutex);
    pthread_cond_destroy(&cond_division_done);
}





//静态线程+信号量同步版本 线程函数
void* threadFunc2(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; ++k)
    {
        sem_wait(&sem_workerstart[t_id]);
        for (int i = k + 1 + t_id; i < N; i += threadnum)
        {
            for (int j = k + 1; j < N; ++j)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
        sem_post(&sem_main); // 唤醒主线程
        sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮

    }
    pthread_exit(NULL);
    return NULL;
}

//静态线程+信号量同步版本
void calculate_2()
{
    sem_init(&sem_main, 0, 0);
    for (int i = 0; i < threadnum; ++i)
    {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    pthread_t handles[threadnum];
    threadParam_t param[threadnum];
    for (int t_id = 0; t_id < threadnum; t_id++)
    {param[t_id].t_id = t_id;
    pthread_create(&handles[t_id], NULL, threadFunc2, &param[t_id]);
    }
    for (int k = 0; k < N; ++k)
    {
        for (int j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        for (int t_id = 0; t_id < threadnum; ++t_id)
        {
            sem_post(&sem_workerstart[t_id]);
        }
        // 主线程睡眠（等待所有的工作线程完成此轮消去任务）
        for(int t_id = 0; t_id < threadnum; ++t_id) 
        { sem_wait(&sem_main);
        }
        // 主线程再次唤醒工作线程进入下一轮次的消去任务
        for (int t_id = 0; t_id < threadnum; ++t_id)
        {
            sem_post(&sem_workerend[t_id]);
        }
    }
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    sem_destroy(&sem_main);
    for (int i = 0; i < threadnum; ++i) {
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
    }
}

//静态线程 + 信号量同步版本 + 三重循环全部纳入线程函数:线程函数（块划分）
void* threadFunc3_1(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; ++k) {
        if (t_id == 0) {
            for (int j = k + 1; j < N; j++) {
                matrix[k][j] = matrix[k][j] / matrix[k][k];
            }
            matrix[k][k] = 1.0;
        }
        else {
            sem_wait(&sem_Divsion1[t_id - 1]);
        }

        if (t_id == 0) {
            for (int i = 0; i < threadnum - 1; ++i) {
                sem_post(&sem_Divsion1[i]);
            }
        }

        int L2 = ceil((N - k) * 1.0 / (threadnum - 1));
        // 循环划分任务
        for (int i = k + (t_id - 1) * L2 + 1; i < N && i < k + t_id * L2 + 1; i++)
        {
            for (int j = k + 1; j < N; ++j) 
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0.0;
        }

        if (t_id == 0) {
            for (int i = 0; i < threadnum - 1; ++i) {
                sem_wait(&sem_leader);
            }

            for (int i = 0; i < threadnum - 1; ++i) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }

    pthread_exit(NULL);
    return NULL;
}

//静态线程 + 信号量同步版本 + 三重循环全部纳入线程函数(块划分方法)
void calculate_3_1()
{//初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < threadnum - 1; ++i)
    {
        sem_init(&sem_Divsion1[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    //创建线程
    pthread_t handles[threadnum];
    threadParam_t param[threadnum];
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc3_1, &param[t_id]);
    }
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    sem_destroy(&sem_leader);
    for (int i = 0; i < threadnum - 1; ++i) {
        sem_destroy(&sem_Divsion1[i]);
        sem_destroy(&sem_Elimination[i]);
    }
}

//静态线程 + 信号量同步版本 + 三重循环全部纳入线程函数:线程函数（列划分方法）
void* threadFunc3_2(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; ++k) {
        if (t_id == 0) {
            for (int j = k + 1; j < N; j++) {
                matrix[k][j] = matrix[k][j] / matrix[k][k];
            }
            matrix[k][k] = 1.0;
        }
        else {
            sem_wait(&sem_Divsion1[t_id - 1]);
        }

        if (t_id == 0) {
            for (int i = 0; i < threadnum - 1; ++i) {
                sem_post(&sem_Divsion1[i]);
            }
        }

        int L3 = (N - (k + 1) + threadnum - 1) / threadnum; // 每个线程处理的列数
        int start_col = k + 1 + t_id * L3;
        int end_col = k + 1 + (t_id + 1) * L3;
        if (end_col >N) end_col = N;  // 防止越界

        // 循环划分任务
        for (int i = k + 1; i < N; i++) {
            for (int j = start_col; j < end_col; ++j) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            if (start_col <= k && k < end_col) {
                matrix[i][k] = 0.0;  // 只有处理该列的线程将此元素置零
            }
        }


        if (t_id == 0) {
            for (int i = 0; i < threadnum - 1; ++i) {
                sem_wait(&sem_leader);
            }

            for (int i = 0; i < threadnum - 1; ++i) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }

    pthread_exit(NULL);
    return NULL;
}

//静态线程 + 信号量同步版本 + 三重循环全部纳入线程函数(列划分方法)
void calculate_3_2()
{//初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < threadnum - 1; ++i)
    {
        sem_init(&sem_Divsion1[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    //创建线程
    pthread_t handles[threadnum];
    threadParam_t param[threadnum];
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc3_2, &param[t_id]);
    }
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    sem_destroy(&sem_leader);
    for (int i = 0; i < threadnum - 1; ++i) {
        sem_destroy(&sem_Divsion1[i]);
        sem_destroy(&sem_Elimination[i]);
    }
}

//静态线程 + 信号量同步版本 + 三重循环全部纳入+多除法工作线程线程函数:线程函数
void* threadFunc3_work(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k <N; ++k) {
        // Calculate each thread's part of the division
        int L4 = (N - (k + 1) + threadnum - 1) / threadnum; // Number of columns per thread
        int start_col = k + 1 + t_id * L4;
        int end_col = start_col + L4;
        if (end_col > N || t_id ==threadnum- 1) end_col = N;

        // Perform the division operation
        for (int j = start_col; j < end_col; j++) {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        if (t_id == 0) {
            matrix[k][k] = 1.0;
        }

        if (t_id == 0) {
            for (int i = 0; i < threadnum - 1; ++i) {
                sem_post(&sem_Divsion1[i]);
            }
        }
        else {
          //  sem_post(&sem_leader); // Signal the main thread
            sem_wait(&sem_Divsion1[t_id - 1]);
        }

        for (int i = k + 1 + t_id; i < N; i += threadnum) {
            for (int j = k + 1; j < N; ++j) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0.0;
        }

        if (t_id == 0) {
            for (int i = 0; i < threadnum - 1; ++i) {
                sem_wait(&sem_leader);
            }

            for (int i = 0; i < threadnum - 1; ++i) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }

    pthread_exit(NULL);
return NULL;
}

//静态线程 + 信号量同步版本 + 三重循环全部纳入+多除法工作线程 线程函数
void calculate_3_work()
{//初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < threadnum - 1; ++i)
    {
        sem_init(&sem_Divsion1[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    //创建线程
    pthread_t handles[threadnum];
    threadParam_t param[threadnum];
    for (int t_id = 0; t_id < threadnum; t_id++) 
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc3_work, &param[t_id]);
    }
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    sem_destroy(&sem_leader);
    for (int i = 0; i < threadnum- 1; ++i) {
        sem_destroy(&sem_Divsion1[i]);
        sem_destroy(&sem_Elimination[i]);
    }
}

//静态线程 + 信号量同步版本 + 三重循环全部纳入线程函数:线程函数
void* threadFunc3(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; ++k) {
        if (t_id == 0) {
            for (int j = k + 1; j < N; j++) {
                matrix[k][j] = matrix[k][j] / matrix[k][k];
            }
            matrix[k][k] = 1.0;
        }
        else {
            sem_wait(&sem_Divsion1[t_id - 1]);
        }

        if (t_id == 0) {
            for (int i = 0; i < threadnum - 1; ++i) {
                sem_post(&sem_Divsion1[i]);
            }
        }

        for (int i = k + 1 + t_id; i < N; i += threadnum) {
            for (int j = k + 1; j < N; ++j) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0.0;
        }

        if (t_id == 0) {
            for (int i = 0; i < threadnum - 1; ++i) {
                sem_wait(&sem_leader);
            }

            for (int i = 0; i < threadnum - 1; ++i) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }

    pthread_exit(NULL);
    return NULL;
}

//静态线程 + 信号量同步版本 + 三重循环全部纳入 线程函数
void calculate_3()
{//初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < threadnum - 1; ++i)
    {
        sem_init(&sem_Divsion1[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    //创建线程
    pthread_t handles[threadnum];
    threadParam_t param[threadnum];
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc3, &param[t_id]);
    }
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    sem_destroy(&sem_leader);
    for (int i = 0; i < threadnum - 1; ++i) {
        sem_destroy(&sem_Divsion1[i]);
        sem_destroy(&sem_Elimination[i]);
    }
}


//静态线程 +barrier 同步线程函数
void* threadFunc4(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; ++k)
    {
        if (t_id == 0)
        {
        for (int j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1.0;}
  pthread_barrier_wait(&barrier_Divsion);
    for (int i = k + 1 + t_id; i < N; i += threadnum)
    {
        for (int j = k + 1; j < N; ++j)
        {
            matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
        }
        matrix[i][k] = 0.0;
    }
    // 第二个同步点
     pthread_barrier_wait(&barrier_Elimination);

    }
    pthread_exit(NULL);
    return NULL;

}

//静态线程 +barrier 同步
void calculate_4()
{
    pthread_barrier_init(&barrier_Divsion, NULL, threadnum);
    pthread_barrier_init(&barrier_Elimination, NULL, threadnum);
    pthread_t handles[threadnum];
    threadParam_t param[threadnum];
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc4, &param[t_id]);
    }
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);

}

//======================================================================NEON========================================================================================================
void calculate_neon()
{
    for (int k = 0; k < N; k++)
    {
        float32x4_t Akk = vmovq_n_f32(matrix[k][k]);
        int j;
        for (j = k + 1; j + 3 < N; j += 4)
        {
            float32x4_t Akj = vld1q_f32(matrix[k] + j);
            Akj = vdivq_f32(Akj, Akk);
            vst1q_f32(matrix[k] + j, Akj);
        }
        for (; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            float32x4_t Aik = vmovq_n_f32(matrix[i][k]);
            for (j = k + 1; j + 3 < N; j += 4)
            {
                float32x4_t Akj = vld1q_f32(matrix[k] + j);
                float32x4_t Aij = vld1q_f32(matrix[i] + j);
                float32x4_t AikMulAkj = vmulq_f32(Aik, Akj);
                Aij = vsubq_f32(Aij, AikMulAkj);
                vst1q_f32(matrix[i] + j, Aij);
            }
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

//neon动态线程函数
void* threadFunc1_neon(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    int t_id = p->t_id;
    int i = k + t_id + 1;

    int j;

    float32x4_t Aik = vmovq_n_f32(matrix[i][k]);
    for (j = k + 1; j + 3 < N; j += 4)
    {
        float32x4_t Akj = vld1q_f32(matrix[k] + j);
        float32x4_t Aij = vld1q_f32(matrix[i] + j);
        float32x4_t AikMulAkj = vmulq_f32(Aik, Akj);
        Aij = vsubq_f32(Aij, AikMulAkj);
        vst1q_f32(matrix[i] + j, Aij);

        //matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
    }
    for (; j < N; j++)
    {
        matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
    }

    matrix[i][k] = 0;
    pthread_exit(NULL);
    return NULL;
}

//neon动态线程版本
void serial_neon()
{
    for (int k = 0; k < N; k++)
    {
        float32x4_t Akk = vmovq_n_f32(matrix[k][k]);
        int j;
        for (j = k + 1; j + 3 < N; j += 4)
        {
            float32x4_t Akj = vld1q_f32(matrix[k] + j);
            Akj = vdivq_f32(Akj, Akk);
            vst1q_f32(matrix[k] + j, Akj);

            //matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        // 串行处理结尾
        for (; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }

        matrix[k][k] = 1;

        int remaining_elements = N - 1 - k;
        int worker_count = remaining_elements / 4;
        pthread_t* handles = (pthread_t*)malloc(worker_count * sizeof(pthread_t)); // 创建对应的 Handles
        threadParam_t* param = (threadParam_t*)malloc(worker_count * sizeof(threadParam_t)); // 创建对应的线程数据结构

        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_create(&handles[t_id], NULL, threadFunc1_neon, (void*)&param[t_id]);
        }
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            pthread_join(handles[t_id], NULL);
        }
        free(handles);
        free(param);
    }

}

//neon静态线程+信号量同步版本线程函数
void* threadFunc2_neon(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; ++k)
    {
        sem_wait(&sem_workerstart[t_id]);


        for (int i = k + 1 + t_id; i < N; i += threadnum)
        {
            float32x4_t Aik = vmovq_n_f32(matrix[i][k]);
            int j;
            for (j = k + 1; j + 3 < N; j += 4)
            {
                float32x4_t Akj = vld1q_f32(matrix[k] + j);
                float32x4_t Aij = vld1q_f32(matrix[i] + j);
                float32x4_t AikMulAkj = vmulq_f32(Aik, Akj);
                Aij = vsubq_f32(Aij, AikMulAkj);
                vst1q_f32(matrix[i] + j, Aij);

                //matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }

        sem_post(&sem_main); // 唤醒主线程
        sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮

    }
    pthread_exit(NULL);
    return NULL;
}

//neon:静态线程+信号量同步版本
void calculate_2_neon()
{
    sem_init(&sem_main, 0, 0);
    for (int i = 0; i < threadnum; ++i)
    {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    pthread_t handles[threadnum];
    threadParam_t param[threadnum];
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc2_neon, &param[t_id]);
    }
    for (int k = 0; k < N; ++k)
    {

        // matrix[k][j] = matrix[k][j] / matrix[k][k];

        float32x4_t Akk = vmovq_n_f32(matrix[k][k]);
        int j;
        for (j = k + 1; j + 3 < N; j += 4)
        {
            float32x4_t Akj = vld1q_f32(matrix[k] + j);
            Akj = vdivq_f32(Akj, Akk);
            vst1q_f32(matrix[k] + j, Akj);
        }
        for (; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }

        matrix[k][k] = 1;

        for (int t_id = 0; t_id < threadnum; ++t_id)
        {
            sem_post(&sem_workerstart[t_id]);
        }
        // 主线程睡眠（等待所有的工作线程完成此轮消去任务）
        for (int t_id = 0; t_id < threadnum; ++t_id)
        {
            sem_wait(&sem_main);
        }
        // 主线程再次唤醒工作线程进入下一轮次的消去任务
        for (int t_id = 0; t_id < threadnum; ++t_id)
        {
            sem_post(&sem_workerend[t_id]);
        }
    }
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    sem_destroy(&sem_main);
    for (int i = 0; i < threadnum; ++i) {
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
    }
}

//neon:静态线程 + 信号量同步版本 + 三重循环全部纳入线程函数:线程函数
void* threadFunc3_neon(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; ++k) {
        if (t_id == 0) {

            float32x4_t Akk = vmovq_n_f32(matrix[k][k]);
            int j;
            for (j = k + 1; j + 3 < N; j += 4)
            {
                float32x4_t Akj = vld1q_f32(matrix[k] + j);
                Akj = vdivq_f32(Akj, Akk);
                vst1q_f32(matrix[k] + j, Akj);

                //matrix[k][j] = matrix[k][j] / matrix[k][k];
            }
            for (; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / matrix[k][k];
            }

            matrix[k][k] = 1;
        }
        else {
            sem_wait(&sem_Divsion1[t_id - 1]);
        }

        if (t_id == 0) {
            for (int i = 0; i < threadnum - 1; ++i) {
                sem_post(&sem_Divsion1[i]);
            }
        }
        for (int i = k + 1 + t_id; i < N; i += threadnum) {
            float32x4_t Aik = vmovq_n_f32(matrix[i][k]);
            int j;
            for (j = k + 1; j + 3 < N; j += 4)
            {
                float32x4_t Akj = vld1q_f32(matrix[k] + j);
                float32x4_t Aij = vld1q_f32(matrix[i] + j);
                float32x4_t AikMulAkj = vmulq_f32(Aik, Akj);
                Aij = vsubq_f32(Aij, AikMulAkj);
                vst1q_f32(matrix[i] + j, Aij);
                //matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }

        if (t_id == 0) {
            for (int i = 0; i < threadnum - 1; ++i) {
                sem_wait(&sem_leader);
            }

            for (int i = 0; i < threadnum - 1; ++i) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }

    pthread_exit(NULL);
    return NULL;
}

//neon:静态线程 + 信号量同步版本 + 三重循环全部纳入线程函数
void calculate_3_neon()
{//初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < threadnum - 1; ++i)
    {
        sem_init(&sem_Divsion1[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    //创建线程
    pthread_t handles[threadnum];
    threadParam_t param[threadnum];
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc3_neon, &param[t_id]);
    }
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    sem_destroy(&sem_leader);
    for (int i = 0; i < threadnum - 1; ++i) {
        sem_destroy(&sem_Divsion1[i]);
        sem_destroy(&sem_Elimination[i]);
    }
}

void* threadFunc4_neon(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; ++k)
    {
        if (t_id == 0)
        {
            float32x4_t Akk = vmovq_n_f32(matrix[k][k]);
            int j;
            for (j = k + 1; j + 3 < N; j += 4)
            {
                float32x4_t Akj = vld1q_f32(matrix[k] + j);
                Akj = vdivq_f32(Akj, Akk);
                vst1q_f32(matrix[k] + j, Akj);
                //matrix[k][j] = matrix[k][j] / matrix[k][k];
            }
            for (; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / matrix[k][k];
            }

            matrix[k][k] = 1;
        }
        pthread_barrier_wait(&barrier_Divsion);
        for (int i = k + 1 + t_id; i < N; i += threadnum)
        {
            float32x4_t Aik = vmovq_n_f32(matrix[i][k]);
            int j;
            for (j = k + 1; j + 3 < N; j += 4)
            {
                 float32x4_t Akj = vld1q_f32(matrix[k] + j);
                float32x4_t Aij = vld1q_f32(matrix[i] + j);
                float32x4_t AikMulAkj = vmulq_f32(Aik, Akj);
                Aij = vsubq_f32(Aij, AikMulAkj);
                vst1q_f32(matrix[i] + j, Aij);

                //matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
        // 第二个同步点
        pthread_barrier_wait(&barrier_Elimination);

    }
    pthread_exit(NULL);
    return NULL;

}

void calculate_4_neon()
{
    pthread_barrier_init(&barrier_Divsion, NULL, threadnum);
    pthread_barrier_init(&barrier_Elimination, NULL, threadnum);
    pthread_t handles[threadnum];
    threadParam_t param[threadnum];
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc4_neon, &param[t_id]);
    }
    for (int t_id = 0; t_id < threadnum; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);

}



// =================================================================================================================================================================================
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// =================================================================================================================================================================================
// 打印矩阵
void print_matrix()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}


//_0：循环划分:这种方法将矩阵的行以循环的方式分配给不同的线程。每个线程负责处理从其起始行开始，步长为总线程数的所有行。
//_1：块划分:在块划分中，矩阵的行被分成连续的段（块），每个线程处理其中一个块。这样的分配通常是为了尽可能均衡地分配工作量，减少线程间同步的需要。
//_2：列划分

int main()
{
    struct timeval start;
    struct timeval end;
    float time = 0;
    init_data();

    cout <<"问题规模"<< N << endl;
    cout  << threadnum << endl;
    // ================================================================================================== serial ============================================================================================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_serial();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "serial:" << time / LOOP << "ms" << endl;

    // ====================================== 动态serial ======================================
  /* for (int i = 0; i < LOOP; i++)
    {

        init_matrix();
         gettimeofday(&start, NULL);
        serial();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
        
    }
    cout << "动态serial:" << time / LOOP  << "ms" << endl;
*/
    // ====================================== 静态+信号量同步serial ======================================
    time = 0;
   for (int i = 0; i < LOOP; i++)
   {

       init_matrix();
       gettimeofday(&start, NULL);
       calculate_2();
       gettimeofday(&end, NULL);
       time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
      
   }
   cout << "静态+信号量同步serial:" << time / LOOP << "ms" << endl;


   // ====================================== 静态+互斥量+ 三重循环全部纳入线程函数serial ======================================
   time = 0;
  for (int i = 0; i < LOOP; i++)
   {

       init_matrix();
       gettimeofday(&start, NULL);
       calculate_mutex();
       gettimeofday(&end, NULL);
       time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;

   }
   cout << "静态+互斥量+ 三重循环全部纳入线程 函数serial:" << time / LOOP << "ms" << endl;
  

   // ====================================== 静态+信号量同步+ 三重循环全部纳入线程函数serial ======================================
   time = 0;
   for (int i = 0; i < LOOP; i++)
   {

       init_matrix();
       gettimeofday(&start, NULL);
       calculate_3();
       gettimeofday(&end, NULL);
       time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;

   }
   cout << "静态+信号量同步+ 三重循环全部纳入线程函数serial:" << time / LOOP << "ms" << endl;


   // ====================================== 静态+信号量同步+ 三重循环全部纳入线程+多个除法工作线程 函数serial ======================================
   time = 0;
   for (int i = 0; i < LOOP; i++)
   {

       init_matrix();
       gettimeofday(&start, NULL);
       calculate_3_work();
       gettimeofday(&end, NULL);
       time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;

   }
   cout << "静态+信号量+ 三重循环+多个除法工作线程 函数serial:" << time / LOOP << "ms" << endl;


   // ====================================== 静态+信号量同步+ 三重循环全部纳入线程函数+块划分方式serial ======================================
   time = 0;
   for (int i = 0; i < LOOP; i++)
   {

       init_matrix();
       gettimeofday(&start, NULL);
       calculate_3_1();
       //QueryPerformanceCounter((LARGE_INTEGER*)&tail);
       gettimeofday(&end, NULL);
       time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
   }
   cout << "静态+信号量同步+ 三重循环全部纳入线程函数+块划分 serial:" << time / LOOP << "ms" << endl;

   // ====================================== 静态+信号量同步+ 三重循环全部纳入线程函数+列划分方式serial ======================================
   time = 0;
   for (int i = 0; i < LOOP; i++)
   {

       init_matrix();
       gettimeofday(&start, NULL);
       calculate_3_2();
       gettimeofday(&end, NULL);
       time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;

   }
   cout << "静态+信号量同步+ 三重循环全部纳入线程函数+列划分 serial:" << time / LOOP << "ms" << endl;

   // ====================================== 静态线程 +barrier 同步serial ======================================
   time = 0;
   for (int i = 0; i < LOOP; i++)
   {

       init_matrix();
       gettimeofday(&start, NULL);
       calculate_4();
       gettimeofday(&end, NULL);
       time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
   }
   cout << "静态线程 +barrier 同步serial:" << time / LOOP << "ms" << endl;

   cout << endl;
    
    //==============================================================================================NEON==========================================================================================================

      // ====================================== NEON ======================================
   time = 0;
   for (int i = 0; i < LOOP; i++)
   {

       init_matrix();
       gettimeofday(&start, NULL);
       calculate_neon();
       gettimeofday(&end, NULL);
       time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
   }
   cout << "neon:" << time / LOOP << "ms" << endl;

   cout << endl;

   // ====================================== 动态 NEON ======================================
  /* time = 0;
   for (int i = 0; i < LOOP; i++)
   {

       init_matrix();
       gettimeofday(&start, NULL);
       serial_neon();
       gettimeofday(&end, NULL);
       time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
   }
   cout << "动态neon:" << time / LOOP << "ms" << endl;

*/

   // ====================================== 静态+信号量 NEON ======================================
   time = 0;
   for (int i = 0; i < LOOP; i++)
   {

       init_matrix();
       gettimeofday(&start, NULL);
       calculate_2_neon();
       gettimeofday(&end, NULL);
       time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
   }
   cout << "静态+信号量neon:" << time / LOOP << "ms" << endl;
   // ====================================== 静态+信号量 NEON ======================================
   time = 0;
   for (int i = 0; i < LOOP; i++)
   {

       init_matrix();
       gettimeofday(&start, NULL);
       calculate_3_neon();
       gettimeofday(&end, NULL);
       time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
   }
   cout << "静态+信号量+三重neon:" << time / LOOP << "ms" << endl;

   // ====================================== 静态+信号量 NEON ======================================
   time = 0;
   for (int i = 0; i < LOOP; i++)
   {

       init_matrix();
       gettimeofday(&start, NULL);
       calculate_4_neon();
       gettimeofday(&end, NULL);
       time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
   }
   cout << "静态+barrierneon:" << time / LOOP << "ms" << endl;
   cout << endl;


    system("pause");
}