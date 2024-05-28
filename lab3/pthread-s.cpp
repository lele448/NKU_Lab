#include <iostream>
#include <fstream>
#include <string>
#include <pthread.h>
#include <windows.h>
#include <iomanip>
#include <vector>

using namespace std;

int numthread =20; // �߳�����

//�̺߳��������ṹ��:
typedef struct {
    int t_id; // �߳� id
    int* EE;  // ����Ԫ��
    int* ER;  // ��Ԫ��
    int COL;  // ���������
    int eeROW; // ����Ԫ�еĸ���
    int erROW; // ��Ԫ�ӵĸ���
    int* flag; // ��Ǹ����׷�����Ԫ�Ӵ������
    int N;    // ÿ�е�int��
    int startRow; // ��ʼ��
    int endRow;   // ������
} threadParam_t;

pthread_mutex_t mutex;

void* threadFunc1(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    int* EE = p->EE;
    int* ER = p->ER;
    int COL = p->COL;
    int eeROW = p->eeROW;
    int erROW = p->erROW;
    int* flag = p->flag;
    int N = p->N;
    int startRow = p->startRow;
    int endRow = p->endRow;

    for (int i = startRow; i < endRow; ++i) {
        int byte = 0;
        int bit = 0;
        while (true) {
            while (byte < N&& EE[i * N + byte] == 0) {
                byte++;
                bit = 0;
            }
            if (byte >= N) {
                break;
            }
            int temp = EE[i * N + byte] << bit;
            while (temp >= 0) {
                bit++;
                temp <<= 1;
            }
            pthread_mutex_lock(&mutex);
            int& isExist = flag[COL - 1 - (byte << 5) - bit];
            if (isExist != 0) {
                int* er = isExist > 0 ? &ER[(isExist - 1) * N] : &EE[~isExist * N];
                pthread_mutex_unlock(&mutex);
                for (int j = 0; j < N; j++) {
                    EE[i * N + j] ^= er[j];
                }
            }
            else {
                isExist = ~i;
                pthread_mutex_unlock(&mutex);
                break;
            }
        }
    }
    pthread_exit(NULL);
    return NULL;
}

// ���߳��㷨:
bool Pthread1(int selection) {
    // selection ������ȡ�ĸ��ļ�
    string Folders[] = { "1_130_22_8", "2_254_106_53", "3_562_170_53", "4_1011_539_263", "5_2362_1226_453",
 "6_3799_2759_1953","7_8399_6375_4535", "8_23045_18748_14325","9_37960_29304_14921","10_43577_39477_54274","11_85401_5724_756" };
    struct Size {
        int a;
        int b;
        int c; // �ֱ�Ϊ������������Ԫ�Ӹ����ͱ���Ԫ�и���
    } fileSize[] = { {130, 22, 8}, {254, 106, 53}, {562, 170, 53}, {1011, 539, 262}, {2362, 1226, 453},
    {3799, 2759, 1953},{8399, 6375, 4535},{23045,18748,14325},{37960,29304,14921},{43577,39477,54274},{85401,5724,756} };

    ifstream erFile;
    ifstream eeFile;
    erFile.open(Folders[selection] + "/1.txt", std::ios::binary); // ��Ԫ���ļ�
    eeFile.open(Folders[selection] + "/2.txt", std::ios::binary); // ����Ԫ���ļ�
    ofstream resFile(Folders[selection] + "/result.txt", ios::trunc); // �����д�ļ�

    int COL = fileSize[selection].a;
    int erROW = fileSize[selection].b;
    int eeROW = fileSize[selection].c;
    int N = (COL + 31) / 32;

    vector<int> ER(erROW * N, 0);
    vector<int> EE(eeROW * N, 0);
    vector<int> flag(COL, 0);

    // ��ȡ��Ԫ��:
    for (int i = 0; i < erROW; i++) {
        int col;
        char ch = ' ';
        erFile >> col;
        int r = COL - 1 - col;
        ER[i * N + (r >> 5)] = 1 << (31 - (r & 31));
        erFile.get(ch);
        flag[col] = i + 1;
        while (erFile.peek() != '\r' && erFile.peek() != '\n') {
            erFile >> col;
            int diff = COL - 1 - col;
            ER[i * N + (diff >> 5)] += 1 << (31 - (diff & 31));
            erFile.get(ch);
        }
    }

    // ��ȡ����Ԫ��:
    for (int i = 0; i < eeROW; i++) {
        int col;
        char ch = ' ';
        while (eeFile.peek() != '\r' && eeFile.peek() != '\n') {
            eeFile >> col;
            int diff = COL - 1 - col;
            EE[i * N + (diff >> 5)] += 1 << (31 - (diff & 31));
            eeFile.get(ch);
        }
        eeFile.get(ch);
    }

    pthread_mutex_init(&mutex, NULL);

    pthread_t* handles = new pthread_t[numthread]; // ����Handle
    threadParam_t* param = new threadParam_t[numthread]; // ������Ӧ���߳����ݽṹ

    int rowsPerThread = (eeROW + numthread - 1) / numthread; // ÿ���̴߳��������

    for (int t_id = 0; t_id < numthread; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].EE = EE.data();
        param[t_id].ER = ER.data();
        param[t_id].COL = COL;
        param[t_id].eeROW = eeROW;
        param[t_id].erROW = erROW;
        param[t_id].flag = flag.data();
        param[t_id].N = N;
        param[t_id].startRow = t_id * rowsPerThread;
        param[t_id].endRow = min((t_id + 1) * rowsPerThread, eeROW);
    }

    // �����߳�
    for (int t_id = 0; t_id < numthread; t_id++)
        pthread_create(&handles[t_id], NULL, threadFunc1, (void*)&param[t_id]);

    // ���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
    for (int t_id = 0; t_id < numthread; t_id++)
        pthread_join(handles[t_id], NULL);

    pthread_mutex_destroy(&mutex);

    // ���õ��Ľ��д�ص��ļ���
    for (int i = 0; i < eeROW; i++) {
        int count = COL - 1;
        for (int j = 0; j < N; j++) {
            int dense = EE[i * N + j];
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

    delete[] handles;
    delete[] param;

    return true;
}

void* threadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    int* EE = p->EE;
    int* ER = p->ER;
    int COL = p->COL;
    int eeROW = p->eeROW;
    int erROW = p->erROW;
    int* flag = p->flag;
    int N = p->N;

    for (int i = t_id; i < eeROW; i += numthread) {
        int byte = 0;
        int bit = 0;
        while (true) {
            while (byte < N&& EE[i * N + byte] == 0) {
                byte++;
                bit = 0;
            }
            if (byte >= N) {
                break;
            }
            int temp = EE[i * N + byte] << bit;
            while (temp >= 0) {
                bit++;
                temp <<= 1;
            }
            int& isExist = flag[COL - 1 - (byte << 5) - bit];
            if (!isExist == 0) {
                int* er = isExist > 0 ? &ER[(isExist - 1) * N] : &EE[~isExist * N];
                for (int j = 0; j < N; j++) {
                    EE[i * N + j] ^= er[j];
                }
            }
            else {
                isExist = ~i;
                break;
            }
        }
    }
    pthread_exit(NULL);
    return NULL;
}

//���߳��㷨:
bool Pthread(int selection) {
    //selection ������ȡ�ĸ��ļ�
    string Folders[] = { "1_130_22_8", "2_254_106_53", "3_562_170_53", "4_1011_539_263", "5_2362_1226_453",
 "6_3799_2759_1953","7_8399_6375_4535", "8_23045_18748_14325","9_37960_29304_14921","10_43577_39477_54274","11_85401_5724_756" };
    struct Size {
        int a;
        int b;
        int c; // �ֱ�Ϊ������������Ԫ�Ӹ����ͱ���Ԫ�и���
    } fileSize[] = { {130, 22, 8}, {254, 106, 53}, {562, 170, 53}, {1011, 539, 262}, {2362, 1226, 453},
    {3799, 2759, 1953},{8399, 6375, 4535},{23045,18748,14325},{37960,29304,14921},{43577,39477,54274},{85401,5724,756} };

    ifstream erFile;
    ifstream eeFile;
    erFile.open(Folders[selection] + "/1.txt", std::ios::binary); // ��Ԫ���ļ�
    eeFile.open(Folders[selection] + "/2.txt", std::ios::binary); // ����Ԫ���ļ�
    ofstream resFile(Folders[selection] + "/result.txt", ios::trunc); // �����д�ļ�

    int COL = fileSize[selection].a;
    int erROW = fileSize[selection].b;
    int eeROW = fileSize[selection].c;
    int N = (COL + 31) / 32;

    vector<int> ER(erROW * N, 0);
    vector<int> EE(eeROW * N, 0);
    vector<int> flag(COL, 0);

    //��ȡ��Ԫ��:
    for (int i = 0; i < erROW; i++) {
        int col;
        char ch = ' ';
        erFile >> col;
        int r = COL - 1 - col;
        ER[i * N + (r >> 5)] = 1 << (31 - (r & 31));
        erFile.get(ch);
        flag[col] = i + 1;
        while (erFile.peek() != '\r' && erFile.peek() != '\n') {
            erFile >> col;
            int diff = COL - 1 - col;
            ER[i * N + (diff >> 5)] += 1 << (31 - (diff & 31));
            erFile.get(ch);
        }
    }

    //��ȡ����Ԫ��:
    for (int i = 0; i < eeROW; i++) {
        int col;
        char ch = ' ';
        while (eeFile.peek() != '\r' && eeFile.peek() != '\n') {
            eeFile >> col;
            int diff = COL - 1 - col;
            EE[i * N + (diff >> 5)] += 1 << (31 - (diff & 31));
            eeFile.get(ch);
        }
        eeFile.get(ch);
    }

    pthread_t* handles = new pthread_t[numthread]; // ����Handle
    threadParam_t* param = new threadParam_t[numthread]; // ������Ӧ���߳����ݽṹ

    for (int t_id = 0; t_id < numthread; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].EE = EE.data();
        param[t_id].ER = ER.data();
        param[t_id].COL = COL;
        param[t_id].eeROW = eeROW;
        param[t_id].erROW = erROW;
        param[t_id].flag = flag.data();
        param[t_id].N = N;
    }

    // �����߳�
    for (int t_id = 0; t_id < numthread; t_id++)
        pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);

    // ���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
    for (int t_id = 0; t_id < numthread; t_id++)
        pthread_join(handles[t_id], NULL);

    // ���õ��Ľ��д�ص��ļ���
    for (int i = 0; i < eeROW; i++) {
        int count = COL - 1;
        for (int j = 0; j < N; j++) {
            int dense = EE[i * N + j];
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

    delete[] handles;
    delete[] param;

    return true;
}

//���߳��㷨:
bool Single_thread(int selection) {
    //selection ������ȡ�ĸ��ļ�
    string Folders[] = { "1_130_22_8", "2_254_106_53", "3_562_170_53", "4_1011_539_263", "5_2362_1226_453",
"6_3799_2759_1953","7_8399_6375_4535", "8_23045_18748_14325","9_37960_29304_14921","10_43577_39477_54274","11_85401_5724_756" };
    struct Size {
        int a;
        int b;
        int c; // �ֱ�Ϊ������������Ԫ�Ӹ����ͱ���Ԫ�и���
    } fileSize[] = { {130, 22, 8}, {254, 106, 53}, {562, 170, 53}, {1011, 539, 262}, {2362, 1226, 453},
    {3799, 2759, 1953},{8399, 6375, 4535},{23045,18748,14325},{37960,29304,14921},{43577,39477,54274},{85401,5724,756} };

    ifstream erFile;
    ifstream eeFile;
    erFile.open(Folders[selection] + "/1.txt", std::ios::binary); // ��Ԫ���ļ�
    eeFile.open(Folders[selection] + "/2.txt", std::ios::binary); // ����Ԫ���ļ�
    ofstream resFile(Folders[selection] + "/result.txt", ios::trunc); // �����д�ļ�

    int COL = fileSize[selection].a;
    int erROW = fileSize[selection].b;
    int eeROW = fileSize[selection].c;
    int N = (COL + 31) / 32;

    vector<int> ER(erROW * N, 0);
    vector<int> EE(eeROW * N, 0);
    vector<int> flag(COL, 0);

    //��ȡ��Ԫ��:
    for (int i = 0; i < erROW; i++) {
        int col;
        char ch = ' ';
        erFile >> col;
        int r = COL - 1 - col;
        ER[i * N + (r >> 5)] = 1 << (31 - (r & 31));
        erFile.get(ch);
        flag[col] = i + 1;
        while (erFile.peek() != '\r' && erFile.peek() != '\n') {
            erFile >> col;
            int diff = COL - 1 - col;
            ER[i * N + (diff >> 5)] += 1 << (31 - (diff & 31));
            erFile.get(ch);
        }
    }

    //��ȡ����Ԫ��:
    for (int i = 0; i < eeROW; i++) {
        int col;
        char ch = ' ';
        while (eeFile.peek() != '\r' && eeFile.peek() != '\n') {
            eeFile >> col;
            int diff = COL - 1 - col;
            EE[i * N + (diff >> 5)] += 1 << (31 - (diff & 31));
            eeFile.get(ch);
        }
        eeFile.get(ch);
    }

    for (int i = 0; i < eeROW; i++) {
        int byte = 0;
        int bit = 0;
        while (true) {
            while (byte < N&& EE[i * N + byte] == 0) {
                byte++;
                bit = 0;
            }
            if (byte >= N) {
                break;
            }
            int temp = EE[i * N + byte] << bit;
            while (temp >= 0) {
                bit++;
                temp <<= 1;
            }
            int& isExist = flag[COL - 1 - (byte << 5) - bit];
            if (!isExist == 0) {
                int* er = isExist > 0 ? &ER[(isExist - 1) * N] : &EE[~isExist * N];
                for (int j = 0; j < N; j++) {
                    EE[i * N + j] ^= er[j];
                }
            }
            else {
                isExist = ~i;
                break;
            }
        }
    }

    // ���õ��Ľ��д�ص��ļ���
    for (int i = 0; i < eeROW; i++) {
        int count = COL - 1;
        for (int j = 0; j < N; j++) {
            int dense = EE[i * N + j];
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

    return true;
}

int main() {
    int counter1;
    int counter2; // ���ڼ�¼���̺߳Ͷ��߳��㷨��1����ִ�еĴ���
    long long head, tail, freq;
    long long head2, tail2, freq2;
    long long head3, tail3, freq3;
    // ��ȡ��ʱƵ��
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    cout.flags(ios::left);

    for (int i = 7; i <= 7; i += 1) { // �����ļ�:
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
        cout << "�߳���:" << numthread << endl;
        cout<< time1 << endl;
        cout  << time2 << endl;
      //  cout << time3 << endl;
        cout << endl;
    }
    return 0;
}

