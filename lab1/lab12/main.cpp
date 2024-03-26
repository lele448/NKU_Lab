#include<iostream>
#include<windows.h>
#include<stdlib.h>
#include <math.h>
using namespace std;
const int n1=21;
const long long n = pow(2, n1) ;

double  sum, a[n],b[n],c[n];


void init(int n)
{
    for (int i = 0; i < n; i++)
    {
a[i] = i;
b[i]=i;
c[i]=i;
    }
}

void ordinary()//平凡算法
{
    long long head, tail, freq;

    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < n; i++)
    {
       sum+=a[i];
    }

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "Col:" << (tail - head) * 1000.0 / freq << "ms" << endl;


void optimize()//两路链式
{
 long long head, tail, freq;
double sum1=0;double sum2=0;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < n; i+=2)
    {
       sum1+=a[i];
       sum2+=a[i+1];
    }
sum=sum1+sum2;
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "Col:" << (tail - head) * 1000.0 / freq << "ms" << endl;

}
void optimize2()//双重循环
{
    long long head, tail, freq;
 QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
 for (int m=n;m>1;m/=2) //log(n)个步骤
for (int i =0; i<m/2; i++)
b[i]=b[i *2]+b[i*2+1];

 QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "Col:" << (tail - head) * 1000.0 / freq << "ms" << endl;
}

void recursion(int n)//递归算法
{
	if (n == 1)return;
	else {
		for (int i = 0; i < (n / 2); i ++) {
			c[i] += c[n - i - 1];
		}
		n = (n + 1) / 2;
		recursion(n);
	}
}


/*
template<int N>
struct ArraySum {

    static double sum(const double* array) {
                return array[N-1] + ArraySum<N-1>::sum(array);
    }
};

template<>
struct ArraySum<0> {
    static double sum(const double*) {
        return double(0);
    }
};

void stylem()
{long long head, tail, freq;

    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

  sum = ArraySum<n>::sum(a);

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "Col:" << (tail - head) * 1000.0 / freq << "ms" << endl;


}


template<int N>
struct SumHelper {
    static void sum(double* a, double& result) {
        result += a[N - 1];
        SumHelper<N - 1>::sum(a, result);
    }
};

template<>
struct SumHelper<0> {
    static void sum(double* a, double& result) {}
};

template<int N>
void sumArray(double* a, double& result) {
    SumHelper<N>::sum(a, result);
}
void stylem() {
    long long head, tail, freq;

    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    sum = 0.0;
    sumArray<n>(a, sum);

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "Col:" << (tail - head) * 1000.0 / freq << "ms" << endl;
}*/
int main()
{
    init(n);
    cout << "n=" << n << endl;
    cout << "ordinary:"; ordinary();

 cout << "optimize1:"; optimize();
cout << "optimize2:"; optimize2();
long long head1,tail1,freq1;
QueryPerformanceFrequency((LARGE_INTEGER*)&freq1);
    QueryPerformanceCounter((LARGE_INTEGER*)&head1);
    recursion(n);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
    cout << "Col:" << (tail1 - head1) * 1000.0 / freq1 << "ms" << endl;
    // cout << "模板:";
    // stylem();

}

