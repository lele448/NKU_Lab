#include <iostream>
#include <fstream>
#include <string>
#include <pthread.h>
#include <windows.h>
#include <iomanip>
using namespace std;



//多线程算法:
bool Pthread(int selection) {
    //selection 决定读取哪个文件
    string Folders[] = { "1_130_22_8", "2_254_106_53", "3_562_170_53", "4_1011_539_263", "5_2362_1226_453",
    "6_3799_2759_1953","7_8399_6375_4535", "8_23045_18748_14325","9_37960_29304_14921","10_43577_39477_54274","11_85401_5724_756" };
    struct Size {
        int a;
        int b;
        int c;//分别为矩阵列数，消元子个数和被消元行个数
    } fileSize[] = { {130, 22, 8}, {254, 106, 53}, {562, 170, 53}, {1011, 539, 262}, {2362, 1226, 453},
    {3799, 2759, 1953},{8399, 6375, 4535},{23045,18748,14325},{37960,29304,14921},{43577,39477,54274},{85401,5724,756} };

    ifstream erFile;
    ifstream eeFile;
    erFile.open( Folders[selection] + "/1.txt", std::ios::binary);//消元子文件
    eeFile.open(Folders[selection] + "/2.txt", std::ios::binary);//被消元行文件
    ofstream resFile(  Folders[selection] + "/result.txt", ios::trunc);//结果回写文件

    int COL = fileSize[selection].a;
    int erROW = fileSize[selection].b;
    int eeROW = fileSize[selection].c;
    int N = (COL + 31) / 32;

    int** ER = new int* [erROW];
    int** EE = new int* [eeROW];
    int* flag = new int[COL] {0};

    //读取消元子:
    for (int i = 0; i < erROW; i++) {
        ER[i] = new int[N] {0};
        int col;
        char ch = ' ';
        erFile >> col;
        int r = COL - 1 - col;
        ER[i][r >> 5] = 1 << (31 - (r & 31));
        erFile.get(ch);
        flag[col] = i + 1;
        while (erFile.peek() != '\r') {
            erFile >> col;
            int diff = COL - 1 - col;
            ER[i][diff >> 5] += 1 << (31 - (diff & 31));
            erFile.get(ch);
        }
    }

    //读取被消元行:
    for (int i = 0; i < eeROW; i++) {
        EE[i] = new int[N] {0};
        int col;
        char ch = ' ';
        while (eeFile.peek() != '\r') {
            eeFile >> col;
            int diff = COL - 1 - col;
            EE[i][diff >> 5] += 1 << (31 - (diff & 31));
            eeFile.get(ch);
        }
        eeFile.get(ch);
    }

    int NUM_THREADS = 20;
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
    for (int i = 0; i < eeROW; i++) {
        int byte = 0;
        int bit = 0;
        int N = (COL + 31) / 32;
        while (true) {
            while (byte < N&& EE[i][byte] == 0) {
                byte++;
                bit = 0;
            }
            if (byte >= N) {
                break;
            }
            int temp = EE[i][byte] << bit;
            while (temp >= 0) {
                bit++;
                temp <<= 1;
            }
            int& isExist = flag[COL - 1 - (byte << 5) - bit];
            if (!isExist == 0) {
                int* er = isExist > 0 ? ER[isExist - 1] : EE[~isExist];
                for (int j = 0; j < N; j++) {
                    EE[i][j] ^= er[j];
                }
            }
            else {
                isExist = ~i;
                break;
            }
        }
    }

    //将得到的结果写回到文件中
    for (int i = 0; i < eeROW; i++) {
        int count = COL - 1;
        for (int j = 0; j < N; j++) {
            int dense = EE[i][j];
            for (int k = 0; k < 32; k++) {
                if (dense == 0) {
                    break;
                }
                else if (dense < 0) {
                    resFile << count - k << ' ';
                }
                dense <<= 1;
            }
            count -= 32;
        }
        resFile << '\n';
    }
    //释放空间:
    for (int i = 0; i < erROW; i++) {
        delete[] ER[i];
    }
    delete[] ER;

    for (int i = 0; i < eeROW; i++) {
        delete[] EE[i];
    }
    delete[] EE;
    delete[] flag;
    return true;
}
//单线程算法:
bool Single_thread(int selection) {
    //selection 决定读取哪个文件
    string Folders[] = { "1_130_22_8", "2_254_106_53", "3_562_170_53", "4_1011_539_263", "5_2362_1226_453",
    "6_3799_2759_1953","7_8399_6375_4535","8_23045_18748_14325","9_37960_29304_14921","10_43577_39477_54274","11_85401_5724_756" };
    struct Size {
        int a;
        int b;
        int c;//分别为矩阵列数，消元子个数和被消元行个数
    } fileSize[] = { {130, 22, 8}, {254, 106, 53}, {562, 170, 53}, {1011, 539, 262}, {2362, 1226, 453},
    {3799, 2759, 1953},{8399, 6375, 4535},{23045,18748,14325},{37960,29304,14921},{43577,39477,54274},{85401,5724,756} };

    ifstream erFile;
    ifstream eeFile;
    erFile.open( Folders[selection] + "/1.txt", std::ios::binary);//消元子文件
    eeFile.open( Folders[selection] + "/2.txt", std::ios::binary);//被消元行文件
    ofstream resFile(Folders[selection] + "/result.txt", ios::trunc);//结果回写文件

    int COL = fileSize[selection].a;
    int erROW = fileSize[selection].b;
    int eeROW = fileSize[selection].c;
    int N = (COL + 31) / 32;

    int** ER = new int* [erROW];
    int** EE = new int* [eeROW];
    int* flag = new int[COL] {0};

    //读取消元子:
    for (int i = 0; i < erROW; i++) {
        ER[i] = new int[N] {0};
        int col;
        char ch = ' ';
        erFile >> col;
        int r = COL - 1 - col;
        ER[i][r >> 5] = 1 << (31 - (r & 31));
        erFile.get(ch);
        flag[col] = i + 1;
        while (erFile.peek() != '\r') {
            erFile >> col;
            int diff = COL - 1 - col;
            ER[i][diff >> 5] += 1 << (31 - (diff & 31));
            erFile.get(ch);
        }
    }

    //读取被消元行:
    for (int i = 0; i < eeROW; i++) {
        EE[i] = new int[N] {0};
        int col;
        char ch = ' ';
        while (eeFile.peek() != '\r') {
            eeFile >> col;
            int diff = COL - 1 - col;
            EE[i][diff >> 5] += 1 << (31 - (diff & 31));
            eeFile.get(ch);
        }
        eeFile.get(ch);
    }

    for (int i = 0; i < eeROW; i++) {
        int byte = 0;
        int bit = 0;
        int N = (COL + 31) / 32;
        while (true) {
            while (byte < N&& EE[i][byte] == 0) {
                byte++;
                bit = 0;
            }
            if (byte >= N) {
                break;
            }
            int temp = EE[i][byte] << bit;
            while (temp >= 0) {
                bit++;
                temp <<= 1;
            }
            int& isExist = flag[COL - 1 - (byte << 5) - bit];
            if (!isExist == 0) {
                int* er = isExist > 0 ? ER[isExist - 1] : EE[~isExist];
                for (int j = 0; j < N; j++) {
                    EE[i][j] ^= er[j];
                }
            }
            else {
                isExist = ~i;
                break;
            }
        }
    }

    //将得到的结果写回到文件中
    for (int i = 0; i < eeROW; i++) {
        int count = COL - 1;
        for (int j = 0; j < N; j++) {
            int dense = EE[i][j];
            for (int k = 0; k < 32; k++) {
                if (dense == 0) {
                    break;
                }
                else if (dense < 0) {
                    resFile << count - k << ' ';
                }
                dense <<= 1;
            }
            count -= 32;
        }
        resFile << '\n';
    }
    //释放空间:
    for (int i = 0; i < erROW; i++) {
        delete[] ER[i];
    }
    delete[] ER;

    for (int i = 0; i < eeROW; i++) {
        delete[] EE[i];
    }
    delete[] EE;
    delete[] flag;
    return true;
}

int main() {

    /*
    int counter1;
    int counter2;
    struct timeval start1;
    struct timeval end1;
    struct timeval start2;
    struct timeval end2;
    cout.flags(ios::left);
    for (int i = 0; i <= 7; i += 1) { //遍历文件:
        //传统算法
        counter1 = 0;
        gettimeofday(&start1, NULL);
        gettimeofday(&end1, NULL);
        while ((end1.tv_sec - start1.tv_sec) < 1) {
            counter1++;
            Single_thread(i);
            gettimeofday(&end1, NULL);
        }

        //多线程算法:
        counter2 = 0;
        gettimeofday(&start2, NULL);
        gettimeofday(&end2, NULL);
        while ((end2.tv_sec - start2.tv_sec) < 1) {
            counter2++;
            Pthread(i);
            gettimeofday(&end2, NULL);
        }

        //用时统计:
        float time1 = (end1.tv_sec - start1.tv_sec) + float((end1.tv_usec - start1.tv_usec)) / 1000000;//单位s;
        float time2 = (end2.tv_sec - start2.tv_sec) + float((end2.tv_usec - start2.tv_usec)) / 1000000;//单位s;

*/
    int counter1;
    int counter2; // 用于记录单线程和多线程算法在1秒内执行的次数
    long long head, tail, freq;
    long long head2, tail2, freq2;
    long long head3, tail3, freq3;
    // 获取计时频率
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    cout.flags(ios::left);

    for (int i = 0; i <= 6; i += 1) { // 遍历文件:
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        Single_thread(i);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);

        float time1 = ((tail - head) * 1000.0 / freq);

        QueryPerformanceFrequency((LARGE_INTEGER*)&freq2);
        QueryPerformanceCounter((LARGE_INTEGER*)&head2);
        Pthread(i);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail2);

        float time2 = ((tail2 - head2) * 1000.0 / freq2);

        cout << fixed << setprecision(6);
        cout << setw(10) << "数据集" << i +1<< ": " << "单线程平均用时：" << setw(20) << time1 << endl;
        cout << setw(10) << " " << "多线程平均用时：" << setw(20) << time2 << endl;
        cout << endl;
        // cout << time1/time2 << endl;
    }
    return 0;
}