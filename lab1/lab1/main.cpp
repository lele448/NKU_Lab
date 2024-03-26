#include<iostream>
#include<windows.h>
#include<stdlib.h>
using namespace std;

const int n=10000;
double b[n][n],col_sum[n],a[n];


 void init(int n)
 {
for(int i=0;i<n;i++)
 {
    a[i]=i;
 for(int j=0;j<n;j++)
    b[i][j]=i+j;
 }}

 void ordinary()
 {
 long long head,tail,freq;

 QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
 QueryPerformanceCounter((LARGE_INTEGER*)&head);
     for(int i=0;i<n;i++)
 {
     col_sum[i]=0.0;
 for(int j=0;j<n;j++)
    col_sum[i]+=b[j][i]*a[j];
 }

 QueryPerformanceCounter((LARGE_INTEGER*)&tail);
 cout<<"Col:"<<(tail-head)*1000.0/freq<<"ms"<<endl;

 }

void optimize()
{
 long long head,tail,freq;
 QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
 QueryPerformanceCounter((LARGE_INTEGER*)&head);
     for(int i=0;i<n;i++)
    col_sum[i]=0.0;
     for(int j=0;j<n;j++)
      for(int i=0;i<n;i++)
      col_sum[i]+=b[j][i]*a[j];
 QueryPerformanceCounter((LARGE_INTEGER*)&tail);
 cout<<"Col:"<<(tail-head)*1000.0/freq<<"ms"<<endl;

}

void unroll()
{
 long long head,tail,freq;

 QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
 QueryPerformanceCounter((LARGE_INTEGER*)&head);
     for(int i=0;i<n;i++)
     col_sum[i]=0.0;
 for(int j=0;j<n;j++)
  {

    for(int i=0;i<n;i+=10)
    {
        col_sum[i]+=b[j][i]*a[j];
        col_sum[i+1]+=b[j][i+1]*a[j];
        col_sum[i+2]+=b[j][i+2]*a[j];
        col_sum[i+3]+=b[j][i+3]*a[j];
        col_sum[i+4]+=b[j][i+4]*a[j];
        col_sum[i+5]+=b[j][i+5]*a[j];
        col_sum[i+6]+=b[j][i+6]*a[j];
        col_sum[i+7]+=b[j][i+7]*a[j];
        col_sum[i+8]+=b[j][i+8]*a[j];
        col_sum[i+9]+=b[j][i+9]*a[j];
    }

  }
 QueryPerformanceCounter((LARGE_INTEGER*)&tail);
 cout<<"Col:"<<(tail-head)*1000.0/freq<<"ms"<<endl;

}

void unroll2()
{
 long long head,tail,freq;
double sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9,sum0;
 QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
 QueryPerformanceCounter((LARGE_INTEGER*)&head);
     for(int i=0;i<n;i++)
     col_sum[i]=0.0;
 for(int j=0;j<n;j+=10)
  {
sum0=0.0;sum1=0.0;sum2=0.0;
sum3=0.0;sum4=0.0;sum5=0.0;
sum6=0.0;sum7=0.0;sum8=0.0;
sum9=0.0;
    for(int i=0;i<n;i++)
    {
        sum0+=b[j][i]*a[j];
        sum1+=b[j+1][i]*a[j+1];
        sum2+=b[j+2][i]*a[j+2];
        sum3+=b[j+3][i]*a[j+3];
        sum4+=b[j+4][i]*a[j+4];
        sum5+=b[j+5][i]*a[j+5];
        sum6+=b[j+6][i]*a[j+6];
        sum7+=b[j+7][i]*a[j+7];
        sum8+=b[j+8][i]*a[j+8];
        sum9+=b[j+9][i]*a[j+9];
    }
    col_sum[j]=sum0;
            col_sum[j+1]=sum1;
            col_sum[j+2]=sum2;
            col_sum[j+3]=sum3;
            col_sum[j+4]=sum4;
            col_sum[j+5]=sum5;
            col_sum[j+6]=sum6;
            col_sum[j+7]=sum7;
            col_sum[j+8]=sum8;
            col_sum[j+9]=sum9;
  }
 QueryPerformanceCounter((LARGE_INTEGER*)&tail);
 cout<<"Col:"<<(tail-head)*1000.0/freq<<"ms"<<endl;

}
int main()
{
    init(n);
    cout<<"n="<<n<<endl;
        cout<<"ordinary:";ordinary();

    cout<<"optimize:";optimize();

    cout<<"unroll:";unroll();
 cout<<"unroll2:";unroll2();


    }

