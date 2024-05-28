#include <iostream>
#include <fstream>
#include <math.h>
#include <windows.h>
#include <xmmintrin.h> // SSE
#include <pthread.h>   // pthread
#include <semaphore.h>
#include <omp.h>

#define _PRINT
// #define _TEST

using namespace std;

const int NUM_THREADS = 8;
// ============================================== pthread �߳̿��Ʊ��� ==============================================
typedef struct
{
    int t_id;
} threadParam_t;

sem_t sem_Division;
pthread_barrier_t barrier;
// ============================================== ������� ==============================================
const int N=100;
const int L = 100;
const int LOOP = 10;
//float** data;
//float** matrix = NULL;
float datanum[N][N];
ofstream res_stream;
float matrix[N][N];

// ��ʼ������
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

// ��data��ʼ��matrix����֤ÿ�ν��м����������һ�µ�
void init_matrix()
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = datanum[i][j];
}

// �����㷨
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

// SSE�����㷨
void calculate_SIMD()
{
    for (int k = 0; k < N; k++)
    {
        // float Akk = matrix[k][k];
        __m128 Akk = _mm_set_ps1(matrix[k][k]);
        int j;
        // ���д���
        for (j = k + 1; j + 3 < N; j += 4)
        {
            // float Akj = matrix[k][j];
            __m128 Akj = _mm_loadu_ps(matrix[k] + j);
            // Akj = Akj / Akk;
            Akj = _mm_div_ps(Akj, Akk);
            // Akj = matrix[k][j];
            _mm_storeu_ps(matrix[k] + j, Akj);
        }
        // ���д����β
        for (; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            // float Aik = matrix[i][k];
            __m128 Aik = _mm_set_ps1(matrix[i][k]);
            for (j = k + 1; j + 3 < N; j += 4)
            {
                // float Akj = matrix[k][j];
                __m128 Akj = _mm_loadu_ps(matrix[k] + j);
                // float Aij = matrix[i][j];
                __m128 Aij = _mm_loadu_ps(matrix[i] + j);
                // AikMulAkj = matrix[i][k] * matrix[k][j];
                __m128 AikMulAkj = _mm_mul_ps(Aik, Akj);
                // Aij = Aij - AikMulAkj;
                Aij = _mm_sub_ps(Aij, AikMulAkj);
                // matrix[i][j] = Aij;
                _mm_storeu_ps(matrix[i] + j, Aij);
            }
            // ���д����β
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// ����ʹ��openmp����simd�Ż�
void calculate_openmp_single_SIMD()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(1) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd aligned(matrix : 16) simdlen(4)
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : guided)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd aligned(matrix : 16) simdlen(4)
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// pthread_discrete �̺߳���
void* threadFunc(void* param)
{
    threadParam_t* thread_param_t = (threadParam_t*)param;
    int t_id = thread_param_t->t_id;
    for (int k = 0; k < N; k++)
    {
        // �����ǰ��0���̣߳�����г��������������̴߳��ڵȴ�״̬
        if (t_id == 0)
        {
            // float Akk = matrix[k][k];
            __m128 Akk = _mm_set_ps1(matrix[k][k]);
            int j;
            //���Ƕ������
            for (j = k + 1; j + 3 < N; j += 4)
            {
                // float Akj = matrix[k][j];
                __m128 Akj = _mm_loadu_ps(matrix[k] + j);
                // Akj = Akj / Akk;
                Akj = _mm_div_ps(Akj, Akk);
                // Akj = matrix[k][j];
                _mm_storeu_ps(matrix[k] + j, Akj);
            }
            for (; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / matrix[k][k];
            }
            matrix[k][k] = 1.0;
        }
        else
        {
            sem_wait(&sem_Division);
        }

        // ����������ɺ������0���̣߳�����Ҫ���������߳�
        if (t_id == 0)
        {
            for (int i = 1; i < NUM_THREADS; i++)
            {
                sem_post(&sem_Division);
            }
        }
        else
        {
            // ѭ����������
            for (int i = k + t_id; i < N; i += (NUM_THREADS - 1))
            {
                // float Aik = matrix[i][k];
                __m128 Aik = _mm_set_ps1(matrix[i][k]);
                int j = k + 1;
                for (; j + 3 < N; j += 4)
                {
                    // float Akj = matrix[k][j];
                    __m128 Akj = _mm_loadu_ps(matrix[k] + j);
                    // float Aij = matrix[i][j];
                    __m128 Aij = _mm_loadu_ps(matrix[i] + j);
                    // AikMulAkj = matrix[i][k] * matrix[k][j];
                    __m128 AikMulAkj = _mm_mul_ps(Aik, Akj);
                    // Aij = Aij - AikMulAkj;
                    Aij = _mm_sub_ps(Aij, AikMulAkj);
                    // matrix[i][j] = Aij;
                    _mm_storeu_ps(matrix[i] + j, Aij);
                }
                for (; j < N; j++)
                {
                    matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
                }
                matrix[i][k] = 0;
            }
        }

        // �����߳�׼��������һ��
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
    return NULL;
}

// pthread_discrete �����㷨
void calculate_pthread()
{
    // �ź�����ʼ��
    sem_init(&sem_Division, 0, 0);
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);

    // �����߳�
    pthread_t threads[NUM_THREADS];
    threadParam_t thread_param_t[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
    {
        thread_param_t[i].t_id = i;
        pthread_create(&threads[i], NULL, threadFunc, (void*)(&thread_param_t[i]));
    }

    // ����ִ���߳�
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // �����ź���
    sem_destroy(&sem_Division);
    pthread_barrier_destroy(&barrier);
}

// ��̬���ݻ���
void calculate_openmp_schedule_static()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : static)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// ��̬���ݻ���
void calculate_openmp_schedule_dynamic()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : dynamic)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// �Զ��������ݻ���
void calculate_openmp_schedule_guided()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : guided)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd aligned(matrix : 16) simdlen(4)
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// ʹ��nowait
void calculate_openmp_schedule_guided_nowait()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : guided) nowait
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// openmp + simd
void calculate_openmp_schedule_guided_SIMD()
{
    int i, j, k;
    float tmp;
    __m128 Akk, Akj, Aik, Aij, AikMulAkj;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp, Akk, Akj, Aik, Aij, AikMulAkj) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
        // float Akk = matrix[k][k];
        Akk = _mm_set_ps1(matrix[k][k]);
        int j;
        // ���д���
        tmp = matrix[k][k];
#pragma omp single
        {
            for (j = k + 1; j + 3 < N; j += 4)
            {
                // float Akj = matrix[k][j];
                Akj = _mm_loadu_ps(matrix[k] + j);
                // Akj = Akj / Akk;
                Akj = _mm_div_ps(Akj, Akk);
                // Akj = matrix[k][j];
                _mm_storeu_ps(matrix[k] + j, Akj);
            }
            // ���д����β
            for (; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1;
        }
#pragma omp for schedule(simd \
                         : guided)
        for (int i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
            // float Aik = matrix[i][k];
            Aik = _mm_set_ps1(matrix[i][k]);
            for (j = k + 1; j + 3 < N; j += 4)
            {
                // float Akj = matrix[k][j];
                Akj = _mm_loadu_ps(matrix[k] + j);
                // float Aij = matrix[i][j];
                Aij = _mm_loadu_ps(matrix[i] + j);
                // AikMulAkj = matrix[i][k] * matrix[k][j];
                AikMulAkj = _mm_mul_ps(Aik, Akj);
                // Aij = Aij - AikMulAkj;
                Aij = _mm_sub_ps(Aij, AikMulAkj);
                // matrix[i][j] = Aij;
                _mm_storeu_ps(matrix[i] + j, Aij);
            }
            // ���д����β
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// ��̬�̰߳汾
void calculate_openmp_static_thread()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : guided)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// ��̬�̰߳汾
void calculate_openmp_dynamic_thread()
{
    int i, j, k;
    float tmp;
    for (k = 0; k < N; k++)
    {
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, tmp) shared(k, matrix, N)
        {
#pragma omp single
            {
                tmp = matrix[k][k];
#pragma omp simd
                for (j = k + 1; j < N; j++)
                {
                    matrix[k][j] = matrix[k][j] / tmp;
                }
                matrix[k][k] = 1.0;
            }
#pragma omp for schedule(simd \
                         : guided)
            for (i = k + 1; i < N; i++)
            {
                tmp = matrix[i][k];
#pragma omp simd
                for (j = k + 1; j < N; j++)
                {
                    matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
                }
                matrix[i][k] = 0;
            }
        }
    }
}

// ���л���
void calculate_openmp_row()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : guided)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// ���л���
void calculate_openmp_column()
{
    int i, j, k;
#pragma omp parallel num_threads(NUM_THREADS), default(none), private(i, j, k), shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp for schedule(simd \
                         : guided)
        for (j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
            for (i = k + 1; i < N; i++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
        }
#pragma omp single
        {
            matrix[k][k] = 1;
            for (i = k + 1; i < N; i++)
            {
                matrix[i][k] = 0;
            }
        }
    }
}


// ���Ժ���
int main()
{
   
    cout << N <<endl;

    long long head, tail, freq;
 
    init_data();
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        calculate_serial();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
       
    }
    cout << "serial:" << ((tail - head) * 1000.0 / freq) / LOOP << "ms" << endl;

    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        calculate_SIMD();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        
    }
    cout << "SIMD:" << ((tail - head) * 1000.0 / freq) / LOOP << "ms" << endl;

    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);

        calculate_openmp_single_SIMD();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    
    }
    cout << "openmp_single_SIMD:" << ((tail - head) * 1000.0 / freq) / LOOP << "ms" << endl;

    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);

        calculate_pthread();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    }
    cout << "pthread:" << ((tail - head) * 1000.0 / freq) / LOOP << "ms" << endl;

    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);

        calculate_openmp_schedule_static();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
      
    }
    cout << "openmp_schedule_static:" << ((tail - head) * 1000.0 / freq) / LOOP << "ms" << endl;

    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);

        calculate_openmp_schedule_dynamic();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    }
    cout << "openmp_schedule_dynamic:" << ((tail - head) * 1000.0 / freq) / LOOP << "ms" << endl;


    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);

        calculate_openmp_schedule_guided();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    }
    cout << "openmp_schedule_guided:" << ((tail - head) * 1000.0 / freq) / LOOP << "ms" << endl;

    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);

        calculate_openmp_schedule_guided_nowait();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        
    }
    cout << "openmp_schedule_guided_nowait:" << ((tail - head) * 1000.0 / freq) / LOOP << "ms" << endl;

    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);

        calculate_openmp_schedule_guided_SIMD();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    }
    cout << "openmp_schedule_guided_SIMD:" << ((tail - head) * 1000.0 / freq) / LOOP << "ms" << endl;
   

   
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);

        calculate_openmp_static_thread();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);

    }
    cout << "openmp_static_thread:" << ((tail - head) * 1000.0 / freq) / LOOP << "ms" << endl;


    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);

        calculate_openmp_dynamic_thread();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
       
    }
    cout << "openmp_dynamic_thread:" << ((tail - head) * 1000.0 / freq) / LOOP << "ms" << endl;
    

    
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);

        calculate_openmp_row();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    }
    cout << "openmp_row:" << ((tail - head) * 1000.0 / freq) / LOOP << "ms" << endl;


   
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);

        calculate_openmp_column();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    }
    cout << "openmp_column:" << ((tail - head) * 1000.0 / freq) / LOOP << "ms" << endl;
    return 0;
}
