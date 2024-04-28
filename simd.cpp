#include <iostream>
#include <fstream>
#include <vector>
#include <windows.h>
#include <ctime>
#include <cmath>
#include <math.h>
#include <string>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <random>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#define MAXSIZE 500

using namespace std;
class index //������
{
public:
	int len = 0;//��ʼ������Ϊ0
	vector<unsigned int> order;//����һ���޷����������������ڴ洢������˳��

};

bool operator < (const index& s1, const index& s2)
{
	return s1.len < s2.len;// ������������������ڱȽ����� index �����ĳ���
}
const int row = 100;
const int listnum = 1000;//���ɵĵ����б�����


const int querynum =300;


class Bitmap
{
public:
	vector<int> bits0;
	vector<int> bits;//һά����,bits��ʾһ�������б��а�����doc���
	vector<int> first_index;
	vector<int> second_index;
	Bitmap(int range)//���캯�� Bitmap(int range)�����ڳ�ʼ��λͼ������ range ��ʾλͼ�ķ�Χ��range ��һ��������������λͼ�����ܰ��������Χ��
	{//λͼ����Ϊ���������洢
	   // this->bits0.resize(range+1);
		this->bits.resize(range / 32 + 1);//�� bits �����Ĵ�С����Ϊ range / 32 + 1
		//��Ϊÿ���������� 32 ������λ��ɵģ����� range / 32 ���Եõ���Ҫ�������������ټ� 1 ��Ϊ��ȷ���ܹ��������п��ܵ���������Ϊ�����Ǵ� 0 ��ʼ�ġ�

		this->first_index.resize(range / 1024 + 1);
		this->second_index.resize(range / 32768 + 1);
	}

	void set_value(int data)//����λͼ�е�ĳ��λ��Ϊ 1������ data ��Ҫ���õ�λ�á�
	{
		int index0 = data / 32;//����dataΪ
		int index1 = index0 / 32;
		int index2 = index1 / 32;

		int tmp0 = data % 32;
		int tmp1 = index0 % 32;
		int tmp2 = index1 % 32;

		//����λ�� data ������� bits��first_index �� second_index �е�������Ȼ�󣬽���Ӧλ�õ�λ����Ϊ 1��
		//this->bits0|= (1 << data);
		this->bits[index0] |= (1 << tmp0);
		this->first_index[index1] |= (1 << tmp1);
		this->second_index[index2] |= (1 << tmp2);
	}

	void reset(int data)//��λͼ�е�ĳ��λ������Ϊ 0
	{//���ȼ������ bits �е�������������Ӧλ�õ�λ����Ϊ 0
		int index = data / 32;
		int tmp = data % 32;
		this->bits[index] &= ~(1 << tmp);//����Ϊ0

		//this->bits &= ~(1 << data);
	}

};





index n_index;

vector<index> idx;

Bitmap S(30000000);//S �� Bitmap ��Ķ������ڱ�ʾλͼ����ʼ����СΪ 30000000��


void search_list_bit_sse1(int* query, vector<index>& idx, int num)
{
	vector<index> t_idx;
	for (int i = 0; i < num; i++) {
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());

	// ����λͼ����
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // ��ʼ��λͼ
		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));  // �ƶ����캯��
	}

	// �󽻼�
	S = bitmap[0];  // �ƶ���ֵ
	for (int i = 1; i < num; i++) {
		for (size_t j = 0; j < S.second_index.size(); j++) {
			if (S.second_index[j] == 0 || (S.second_index[j] & bitmap[i].second_index[j] == 0)) continue;  // ��������Ϊ0��ֱ������
			for (size_t t = j * 32; t < (j + 1) * 32; t += 8) {
				__m256i S_vec = _mm256_loadu_si256((__m256i*) & S.first_index[t]);
				__m256i bitmap_vec = _mm256_loadu_si256((__m256i*) & bitmap[i].first_index[t]);
				__m256i result_vec = _mm256_and_si256(S_vec, bitmap_vec);
				_mm256_storeu_si256((__m256i*) & S.first_index[t], result_vec);
			}
		}
	}
}

void search_list_bit_sse(int* query, vector<index>& idx, int num) {
	vector<index> t_idx;
	for (int i = 0; i < num; i++) {
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());

	// ����λͼ����
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // ��ʼ��λͼ

		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));  // �ƶ����캯��
	}

	// �󽻼�
	Bitmap& S = bitmap[0];

	for (int i = 1; i < num; i++) {
		for (size_t j = 0; j < S.second_index.size(); j += 4) {
			bool judge = false;
			alignas(16) __m128i var0 = _mm_load_si128((__m128i*) & S.second_index[j]);
			alignas(16) __m128i var1 = _mm_load_si128((__m128i*) & bitmap[i].second_index[j]);
			__m128i var = _mm_and_si128(var0, var1);
			if (_mm_movemask_epi8(_mm_cmpeq_epi8(var, _mm_setzero_si128())) != 0xFFFF) {  // ��� var �Ƿ��������Ԫ��
				_mm_store_si128((__m128i*) & S.second_index[j], var);
				// �� first_index �� bits ��һ������
				for (size_t t = j * 32; t < (j + 1) * 32; t += 4) {
					alignas(16) __m128i a0 = _mm_load_si128((__m128i*) & S.first_index[t]);
					alignas(16) __m128i a1 = _mm_load_si128((__m128i*) & bitmap[i].first_index[t]);
					__m128i a = _mm_and_si128(a0, a1);
					if (_mm_movemask_epi8(_mm_cmpeq_epi8(a, _mm_setzero_si128())) != 0xFFFF) {
						_mm_store_si128((__m128i*) & S.first_index[t], a);
						for (size_t k = t * 32; k < (t + 1) * 32; k += 4) {
							alignas(16) __m128i b0 = _mm_load_si128((__m128i*) & S.bits[k]);
							alignas(16) __m128i b1 = _mm_load_si128((__m128i*) & bitmap[i].bits[k]);
							__m128i b = _mm_and_si128(b0, b1);
							if (_mm_movemask_epi8(_mm_cmpeq_epi8(b, _mm_setzero_si128())) != 0xFFFF) {
								_mm_store_si128((__m128i*) & S.bits[k], b);
								judge = true;
							}
						}
					}
				}
			}
			if (!judge) {
				var0 = _mm_setzero_si128();
			}
		}
	}
}
void search_list_bit_sse_non(int* query, vector<index>& idx, int num) {
	vector<index> t_idx;
	for (int i = 0; i < num; i++) {
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());

	// ����λͼ����
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // ��ʼ��λͼ

		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));  // �ƶ����캯��
	}

	// �󽻼�
	Bitmap& S = bitmap[0];

	for (int i = 1; i < num; i++) {
		for (size_t j = 0; j < S.second_index.size(); j += 4) {
			bool judge = false;
			__m128i var0 = _mm_loadu_si128((__m128i*) & S.second_index[j]);
			__m128i var1 = _mm_loadu_si128((__m128i*) & bitmap[i].second_index[j]);
			__m128i var = _mm_and_si128(var0, var1);
			if (_mm_movemask_epi8(_mm_cmpeq_epi8(var, _mm_setzero_si128())) != 0xFFFF) {  // ��� var �Ƿ��������Ԫ��
				_mm_store_si128((__m128i*) & S.second_index[j], var);
				// �� first_index �� bits ��һ������
				for (size_t t = j * 32; t < (j + 1) * 32; t += 4) {
					__m128i a0 = _mm_loadu_si128((__m128i*) & S.first_index[t]);
					__m128i a1 = _mm_loadu_si128((__m128i*) & bitmap[i].first_index[t]);
					__m128i a = _mm_and_si128(a0, a1);
					if (_mm_movemask_epi8(_mm_cmpeq_epi8(a, _mm_setzero_si128())) != 0xFFFF) {
						_mm_store_si128((__m128i*) & S.first_index[t], a);
						for (size_t k = t * 32; k < (t + 1) * 32; k += 4) {
							__m128i b0 = _mm_loadu_si128((__m128i*) & S.bits[k]);
							__m128i b1 = _mm_loadu_si128((__m128i*) & bitmap[i].bits[k]);
							__m128i b = _mm_and_si128(b0, b1);
							if (_mm_movemask_epi8(_mm_cmpeq_epi8(b, _mm_setzero_si128())) != 0xFFFF) {
								_mm_store_si128((__m128i*) & S.bits[k], b);
								judge = true;
							}
						}
					}
				}
			}
			if (!judge) {
				var0 = _mm_setzero_si128();
			}
		}
	}
}
void search_list_bit_avx(int* query, vector<index>& idx, int num) {
	vector<index> t_idx;
	for (int i = 0; i < num; i++) {
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());

	// ����λͼ����
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // ��ʼ��λͼ

		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));  // �ƶ����캯��
	}

	// �󽻼�
	Bitmap& S = bitmap[0];

for (int i = 1; i < num; i++) {
		for (size_t j = 0; j < S.second_index.size()/8; j += 8) {
			bool judge = false;
			alignas(32)__m256i var0 = _mm256_load_si256((__m256i*) & S.second_index[j]);
			alignas(32)__m256i var1 = _mm256_load_si256((__m256i*) & bitmap[i].second_index[j]);
			__m256i var = _mm256_and_si256(var0, var1);
			if (!_mm256_testz_si256(var, var)) {  // ��� var �Ƿ��������Ԫ��
				_mm256_store_si256((__m256i*) & S.second_index[j], var);


				// �� first_index �� bits ��һ������
				for (size_t t = j * 32; t < ((j + 1) * 32)/8; t += 8) {
					alignas(32)__m256i  a0 = _mm256_load_si256((__m256i*) & S.first_index[t]);
					alignas(32)__m256i  a1 = _mm256_load_si256((__m256i*) & bitmap[i].first_index[t]);
					__m256i  a = _mm256_and_si256(a0, a1);
					if (!_mm256_testz_si256(a,a)) {
						_mm256_store_si256((__m256i*) & S.first_index[t], a);


						for (size_t k = t * 32; k < ((t + 1) * 32)/8; k += 8) {
							alignas(32)__m256i b0 = _mm256_load_si256((__m256i*) & S.bits[k]);
							alignas(32)__m256i b1 = _mm256_load_si256((__m256i*) & bitmap[i].bits[k]);
							__m256i b = _mm256_and_si256(b0, b1);

							if (!_mm256_testz_si256(b,b)) {
								_mm256_store_si256((__m256i*) & S.bits[k], b);
								judge = true;
							}
						}
					}
				}
			}
			if (!judge) {
				var0 = _mm256_setzero_si256();

			}
		}
	}
}
void search_list_bit_avx_non(int* query, vector<index>& idx, int num) {
	vector<index> t_idx;
	for (int i = 0; i < num; i++) {
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());

	// ����λͼ����
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // ��ʼ��λͼ

		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));  // �ƶ����캯��
	}

	// �󽻼�
	Bitmap& S = bitmap[0];

	for (int i = 1; i < num; i++) {
		for (size_t j = 0; j < S.second_index.size() / 8; j += 8) {
			bool judge = false;
			__m256i var0 = _mm256_loadu_si256((__m256i*) & S.second_index[j]);
			__m256i var1 = _mm256_loadu_si256((__m256i*) & bitmap[i].second_index[j]);
			__m256i var = _mm256_and_si256(var0, var1);
			if (!_mm256_testz_si256(var, var)) {  // ��� var �Ƿ��������Ԫ��
				_mm256_store_si256((__m256i*) & S.second_index[j], var);


				// �� first_index �� bits ��һ������
				for (size_t t = j * 32; t < ((j + 1) * 32) / 8; t += 8) {
					__m256i  a0 = _mm256_loadu_si256((__m256i*) & S.first_index[t]);
					__m256i  a1 = _mm256_loadu_si256((__m256i*) & bitmap[i].first_index[t]);
					__m256i  a = _mm256_and_si256(a0, a1);
					if (!_mm256_testz_si256(a, a)) {
						_mm256_store_si256((__m256i*) & S.first_index[t], a);


						for (size_t k = t * 32; k < ((t + 1) * 32) / 8; k += 8) {
							__m256i b0 = _mm256_loadu_si256((__m256i*) & S.bits[k]);
							__m256i b1 = _mm256_loadu_si256((__m256i*) & bitmap[i].bits[k]);
							__m256i b = _mm256_and_si256(b0, b1);

							if (!_mm256_testz_si256(b, b)) {
								_mm256_store_si256((__m256i*) & S.bits[k], b);
								judge = true;
							}
						}
					}
				}
			}
			if (!judge) {
				var0 = _mm256_setzero_si256();

			}
		}
	}
}

//����
void search_list_bit(int* query, vector<index>& idx, int num)
{
	vector<index> t_idx;
	for (int i = 0; i < num; i++) {
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());

	// ����λͼ����
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // ��ʼ��λͼ
		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));  // �ƶ����캯��
	}

	// �󽻼�
	S = bitmap[0];  // �ƶ���ֵ
	for (int i = 1; i < num; i++) {
		for (size_t j = 0; j < S.second_index.size(); j++) {
			if (S.second_index[j] == 0 || (S.second_index[j] & bitmap[i].second_index[j] == 0)) continue;  // ��������Ϊ0��ֱ������
			for (size_t t = j * 32; t < (j + 1) * 32; t++) {
				if (S.first_index[t] == 0 || (S.first_index[t] & bitmap[i].first_index[t] == 0)) continue;  // һ������Ϊ0��ֱ������
				for (size_t k = t * 32; k < (t + 1) * 32; k++) {
					if (S.bits[k] == 0) continue;  // λΪ0��ֱ������
					S.bits[k] &= bitmap[i].bits[k];  // ��λ�����
				}
			}
		}
	}
}



//һ��һ��Ԫ�أ�(32��Ԫ��Ϊһ������Ԫ��)
void search_list_bit0(int* query, vector<index>& idx, int num)//������ʹ��λͼ��ʾ�������б��н�������
{
	vector<index> t_idx;//��������������������ڴ洢���ݲ�ѯλ�ô� idx ��ȡ�õ�������Ϣ
	for (int i = 0; i < num; i++)
	{
		t_idx.push_back(idx[query[i]]);//t_idx������е����б������
	}
	sort(t_idx.begin(), t_idx.end());//����
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++)
	{
		bitmap.push_back(30000000);//��ʼ��
		for (int j = 0; j < t_idx[i].len; j++)//jΪһ��t_idx[]�����б�ĳ���
		{
			bitmap[i].set_value(t_idx[i].order[j]);//����bitmap
		}
	}
	S = bitmap[0];
	for (int i = 1; i < num; i++)
	{
		for (int j = 0; j < S.bits.size(); j++)
		{
			bool judge = false;
			if (S.bits[j] != 0)
			{
				S.bits[j] &= bitmap[i].bits[j];
				if (S.bits[j] != 0)//������0��˵��Ϊ1������ͬ
				{
					judge = true;
				}
				if (judge == false)
				{
					S.bits[j] = 0;
				}
			}
		}
	}
}

//�˺�������Ҫ�����Ǹ��ݲ�ѯ������λ�ã�ʹ��λͼ��ʾ�������б����������ͨ��λ�����λͼ���и��£����յõ�ƥ��Ľ��λͼ S��
void search_element_bit(int* query, vector<index>& idx, int num)//��ʹ��λͼ��ʾ�������б���������ͬ��Ԫ��
{
	vector<index> t_idx;
	for (int i = 0; i < num; i++)
	{
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++)
	{
		bitmap.push_back(30000000);
		for (int j = 0; j < t_idx[i].len; j++)
		{
			bitmap[i].set_value(t_idx[i].order[j]);
		}
	}
	bool judge = false;
	S = bitmap[0];


	for (int i = 0; i < S.second_index.size(); i++)
	{
		for (int j = 1; j < num; j++)
		{
			if (S.second_index[i] != 0 && (S.second_index[i] &= bitmap[j].second_index[i] != 0))
			{


				for (int t = i * 32; t < i * 32 + 32; t++)
				{
					if (S.first_index[t] == 0 || (S.first_index[i] &= bitmap[j].first_index[t] == 0))
						continue;
					for (int l = t * 32; l < t * 32 + 32; l++)
					{
						if (S.bits[l] != 0 && (S.bits[l] &= bitmap[j].bits[l] != 0))
							judge = true;

					}
				}

			}
			else
			{
				break;
			}
			if (judge == false)
			{
				S.second_index[i] = 0;
				break;
			}

		}
	}
}

void search_element_bit_sse(int* query, vector<index>& idx, int num) {
	vector<index> t_idx;
	for (int i = 0; i < num; i++) {
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // �����������һ�����캯����ʼ����С
		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));
	}

	bool judge = false;
	Bitmap& S = bitmap[0];

	for (size_t i = 0; i < S.second_index.size(); i += 4) {
		alignas(16)__m128i var0 = _mm_load_si128((__m128i*) & S.second_index[i]);
		for (int j = 1; j < num; j++) {
			alignas(16)__m128i var1 = _mm_load_si128((__m128i*) & bitmap[j].second_index[i]);
			__m128i var = _mm_and_si128(var0, var1);
			int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(var, _mm_setzero_si128()));
			if (mask != 0xFFFF) { // �����ȫΪ0
				var0 = var; // ���� S.second_index ֱ���ڼĴ����н���
				for (int t = i * 32; t < i * 32 + 32; t += 4) {
					alignas(16)__m128i a0 = _mm_load_si128((__m128i*) & S.first_index[t]);
					alignas(16)__m128i a1 = _mm_load_si128((__m128i*) & bitmap[j].first_index[t]);
					__m128i a = _mm_and_si128(a0, a1);
					int first_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(a, _mm_setzero_si128()));
					if (first_mask != 0xFFFF) {
						a0 = a; // ���� S.first_index
						for (int l = t * 32; l < t * 32 + 32; l += 4) {
							alignas(16)__m128i b0 = _mm_load_si128((__m128i*) & S.bits[l]);
							alignas(16)__m128i b1 = _mm_load_si128((__m128i*) & bitmap[j].bits[l]);
							__m128i b = _mm_and_si128(b0, b1);
							int bits_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(b, _mm_setzero_si128()));
							if (bits_mask != 0xFFFF) {
								judge = true; // ���� S.bits
								b0 = b;
							}
						}
					}
				}
			}
			else {
				break;
			}
		}
		if (!judge) {
			_mm_store_si128((__m128i*) & S.second_index[i], _mm_setzero_si128());
		}
	}
}
void search_element_bit_sse_non(int* query, vector<index>& idx, int num) {
	vector<index> t_idx;
	for (int i = 0; i < num; i++) {
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // �����������һ�����캯����ʼ����С
		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));
	}

	bool judge = false;
	Bitmap& S = bitmap[0];

	for (size_t i = 0; i < S.second_index.size(); i += 4) {
		__m128i var0 = _mm_loadu_si128((__m128i*) & S.second_index[i]);
		for (int j = 1; j < num; j++) {
			__m128i var1 = _mm_loadu_si128((__m128i*) & bitmap[j].second_index[i]);
			__m128i var = _mm_and_si128(var0, var1);
			int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(var, _mm_setzero_si128()));
			if (mask != 0xFFFF) { // �����ȫΪ0
				var0 = var; // ���� S.second_index ֱ���ڼĴ����н���
				for (int t = i * 32; t < i * 32 + 32; t += 4) {
					__m128i a0 = _mm_loadu_si128((__m128i*) & S.first_index[t]);
					__m128i a1 = _mm_loadu_si128((__m128i*) & bitmap[j].first_index[t]);
					__m128i a = _mm_and_si128(a0, a1);
					int first_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(a, _mm_setzero_si128()));
					if (first_mask != 0xFFFF) {
						a0 = a; // ���� S.first_index
						for (int l = t * 32; l < t * 32 + 32; l += 4) {
							__m128i b0 = _mm_loadu_si128((__m128i*) & S.bits[l]);
							__m128i b1 = _mm_loadu_si128((__m128i*) & bitmap[j].bits[l]);
							__m128i b = _mm_and_si128(b0, b1);
							int bits_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(b, _mm_setzero_si128()));
							if (bits_mask != 0xFFFF) {
								judge = true; // ���� S.bits
								b0 = b;
							}
						}
					}
				}
			}
			else {
				break;
			}
		}
		if (!judge) {
			_mm_store_si128((__m128i*) & S.second_index[i], _mm_setzero_si128());
		}
	}
}
void search_element_bit_avx(int* query, vector<index>& idx, int num) {
	vector<index> t_idx;
	for (int i = 0; i < num; i++) {
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // �����������һ�����캯����ʼ����С
		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));
	}

	bool judge = false;
	Bitmap& S = bitmap[0];

	for (size_t i = 0; i < S.second_index.size()/8; i += 8) {

		alignas(32)__m256i var0 = _mm256_load_si256((__m256i*) & S.second_index[i]);
		for (int j = 1; j < num; j++) {

			alignas(32)__m256i var1 = _mm256_load_si256((__m256i*) & bitmap[j].second_index[i]);
			__m256i var = _mm256_and_si256(var0, var1);

			if (!_mm256_testz_si256(var, var)) { // �����ȫΪ0
				var0 = var; // ���� S.second_index ֱ���ڼĴ����н���
				for (int t = i * 32; t < (i * 32 + 32)/8; t += 8) {
					alignas(32)__m256i a0 = _mm256_load_si256((__m256i*) & S.first_index[t]);
					alignas(32)__m256i a1 = _mm256_load_si256((__m256i*) & bitmap[j].first_index[t]);

					__m256i  a = _mm256_and_si256(a0, a1);
					if (!_mm256_testz_si256(a, a)) {
						a0 = a; // ���� S.first_index
						for (int l = t * 32; l < (t * 32 + 32)/8; l += 8) {
							alignas(32)__m256i b0 = _mm256_load_si256((__m256i*) & S.bits[l]);
							alignas(32)__m256i b1 = _mm256_load_si256((__m256i*) & bitmap[j].bits[l]);
							__m256i b = _mm256_and_si256(b0, b1);
							
							if (!_mm256_testz_si256(b, b)) {
								judge = true; // ���� S.bits
								b0 = b;
							}
						}
					}
				}
			}
			else {
				break;
			}
		}
		if (!judge) {
			_mm256_store_si256((__m256i*) & S.second_index[i], _mm256_setzero_si256());

		}
	}
}
void search_element_bit_avx_non(int* query, vector<index>& idx, int num) {
	vector<index> t_idx;
	for (int i = 0; i < num; i++) {
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // �����������һ�����캯����ʼ����С
		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));
	}

	bool judge = false;
	Bitmap& S = bitmap[0];

	for (size_t i = 0; i < S.second_index.size() / 8; i += 8) {

		__m256i var0 = _mm256_loadu_si256((__m256i*) & S.second_index[i]);
		for (int j = 1; j < num; j++) {

			__m256i var1 = _mm256_loadu_si256((__m256i*) & bitmap[j].second_index[i]);
			__m256i var = _mm256_and_si256(var0, var1);

			if (!_mm256_testz_si256(var, var)) { // �����ȫΪ0
				var0 = var; // ���� S.second_index ֱ���ڼĴ����н���
				for (int t = i * 32; t < (i * 32 + 32) / 8; t += 8) {
					__m256i a0 = _mm256_loadu_si256((__m256i*) & S.first_index[t]);
					__m256i a1 = _mm256_loadu_si256((__m256i*) & bitmap[j].first_index[t]);

					__m256i  a = _mm256_and_si256(a0, a1);
					if (!_mm256_testz_si256(a, a)) {
						a0 = a; // ���� S.first_index
						for (int l = t * 32; l < (t * 32 + 32) / 8; l += 8) {
							__m256i b0 = _mm256_loadu_si256((__m256i*) & S.bits[l]);
							__m256i b1 = _mm256_loadu_si256((__m256i*) & bitmap[j].bits[l]);
							__m256i b = _mm256_and_si256(b0, b1);

							if (!_mm256_testz_si256(b, b)) {
								judge = true; // ���� S.bits
								b0 = b;
							}
						}
					}
				}
			}
			else {
				break;
			}
		}
		if (!judge) {
			_mm256_store_si256((__m256i*) & S.second_index[i], _mm256_setzero_si256());

		}
	}
}

void search_element_bit0(int* query, vector<index>& idx, int num)//��ʹ��λͼ��ʾ�������б���������ͬ��Ԫ��
{
	vector<index> t_idx;
	for (int i = 0; i < num; i++)
	{
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++)
	{
		bitmap.push_back(30000000);
		for (int j = 0; j < t_idx[i].len; j++)
		{
			bitmap[i].set_value(t_idx[i].order[j]);
		}
	}
	bool judge = false;
	S = bitmap[0];
	for (int i = 0; i < S.bits.size(); i++)
	{
		for (int j = 1; j < num; j++)//�б�����
		{
			if (S.bits[i] == 0 || (S.bits[i] & bitmap[j].bits[i] == 0))
			{
				S.bits[i] = 0; continue;
			}

		}
	}
}
void search_list(int* query, vector<index>& idx, int num) {
	vector<index> t_idx;
	for (int i = 0; i < num; i++) {
		t_idx.push_back(idx[query[i]]);
	}

	sort(t_idx.begin(), t_idx.end());

	// �洢��ǰ����
	vector<unsigned int> intersection = t_idx[0].order;

	for (int i = 1; i < num; i++) {
		// ��ʱ�洢�µĽ���
		vector<unsigned int> new_intersection;

		// ʹ�ö��ֲ����ڵ�ǰ�����в����Ƿ�����ڽ����е�Ԫ��
		for (unsigned int elem : intersection) {
			if (binary_search(t_idx[i].order.begin(), t_idx[i].order.end(), elem)) {
				new_intersection.push_back(elem);
			}
		}

		// ���µ�ǰ����Ϊ�µĽ���
		intersection = new_intersection;
	}

	// ���µ�ǰ�����ĳ���
	n_index.len = intersection.size();
	// ���µ�ǰ������˳��
	n_index.order = move(intersection);
}


void search_element(int* query, vector<index>& idx, int num) {
	vector<index> t_idx;
	for (int i = 0; i < num; i++) {
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());

	// ѡ���������
	index& shortest_list = t_idx[0];
	for (int i = 0; i < shortest_list.len; i++) {
		unsigned int elem = shortest_list.order[i];
		bool found_in_all = true;

		// ��������������ж��ֲ���
		for (int j = 1; j < num; j++) {
			index& current_list = t_idx[j];
			if (!binary_search(current_list.order.begin(), current_list.order.end(), elem)) {
				found_in_all = false;
				break;
			}
		}

		// ���Ԫ�ش��������������У����뽻��
		if (found_in_all) {
			n_index.len++;
			n_index.order.push_back(elem);
		}
	}
}




void gettime(void (*func)(int* query, vector<index>& idx, int num), int t_query[querynum][5], vector<index>& idx)
{

	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);

	for (int i = 0; i < querynum; i++)
	{
		int num = 0;
		for (int j = 0; j < 5; j++)
		{
			if (t_query[i][j] != 0)
			{
				num++;
			}
		}
		int* query = new int[num];
		for (int j = 0; j < num; j++)
		{
			query[j] = t_query[i][j];
		}

		func(query, idx, num);

		delete query;
	}

	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << ((tail - head) * 1000.0 / freq) << "ms" << endl;
}



int main()
{

	fstream outfile;
	outfile.open("ExpIndex", ios::binary | ios::in);
	for (int i = 0; i < 2000; i++)//2000���ĵ�
	{
		index tmp;//ÿ�ε��������ļ��ж�ȡһ�� index ���������,tmp��һ�������б�
		outfile.read((char*)&tmp.len, sizeof(tmp.len));//���ȶ�ȡ len ��Ա������ֵ��Ȼ����� len �Ĵ�С��ȡ��Ӧ������ order ���ݣ��������Ǵ洢�� tmp ��
		for (int j = 0; j < (tmp.len); j++)
		{//n_tmpΪdoc�ĵ�
			unsigned int n_tmp;//�ݴ���ļ��ж�ȡ������
			outfile.read((char*)&n_tmp, sizeof(n_tmp));//���ļ��ж�ȡһ�����ֽ��޷���������С�ˣ���������洢�� n_tmp ��
			tmp.order.push_back(n_tmp);//����ȡ���޷����������� n_tmp ��ӵ� tmp ����� order ������
		}
		idx.push_back(tmp);//����ȡ�� tmp ������ӵ���Ϊ idx �� index ������
		//tmp �����ݴ�һ�������Լ��䳤����Ϣ���� idx ���Ǵ洢�˶�� tmp �����������
	}

	outfile.close();//�ر������ļ��� outfile

	outfile.open("ExpQuery", ios::in);
	//1000�β�ѯ��¼��ÿ�β�ѯ����5�������б�
	int t_query[querynum][5] = { 0 };//����һ����ά�������� t_query[1000][5] ����ʼ��Ϊȫ��
	string line;
	int n_count = 0;

	//���º������ǽ�ExpQuery�ļ��е�����ת��Ϊ����洢

	while (getline(outfile, line) && n_count < querynum)//ʹ��һ��ѭ������ȡ�ļ���ÿһ�У�ÿ�д���һ����ѯ
	{
		stringstream ss(line);//����ÿһ�У�ʹ�� stringstream ���� ss ���зִʣ������е�������ȡ���������洢�� t_query �����С�
		int addr = 0;
		while (!ss.eof())
		{
			int tmp;
			ss >> tmp;
			t_query[n_count][addr] = tmp;//������ n_count ���ڼ�¼��ȡ�Ĳ�ѯ����
			addr++;
		}
		n_count++;
	}
	//t_query�������ǲ�ѯ��¼����idx����ǵ��������洢�˶�� tmp ���������
	outfile.close();
	/*
//
	srand(time(0));
	for (int i = 0; i < listnum; i++)
	{
		index t;// ����500�������б�
		t.len= rand() % (150 - 50 + 1) + 30;//30~80
		vector<int> forRandom;//����������
		for (int j = 0; j < t.len * 4; j++)//Ϊ�˲��ظ�
		{
			forRandom.push_back(j);
		}
		random_shuffle(forRandom.begin(), forRandom.end());//�������

		for (int j = 0; j < t.len; j++)
		{
			int docId = forRandom[j];

			t.order.push_back(docId);
		}
		sort(t.order.begin(), t.order.end());//���ĵ��������

		idx.push_back(t);

	}



int t_query[row][5]={ 0 };
	for (int i = 0; i < row; i++)
		{// ��200�β�ѯ
		//int testQuery[5];
		for (int j = 0; j < 5; j++)
				{//cout<< j << endl;
			t_query[i][j] = rand() % listnum;
	   }
	}*/
	//
	cout << "querynum=" << querynum<<endl;

	cout << "λͼ�洢��ʽ�µİ������󽻣�����������:";
	gettime(search_list_bit, t_query, idx);

	cout << "λͼ�洢��ʽ�µİ������󽻣�sse1��:";
	gettime(search_list_bit_sse1, t_query, idx);
	
	cout << "λͼ�洢��ʽ�µİ������󽻣�sse��:";
	gettime(search_list_bit_sse, t_query, idx);
	cout << "λͼ�洢��ʽ�µİ������󽻣�sse�����룩:";
	gettime(search_list_bit_sse_non, t_query, idx);
	cout << "λͼ�洢��ʽ�µİ������󽻣�avx��:";
	gettime(search_list_bit_avx, t_query, idx);
	cout << "λͼ�洢��ʽ�µİ������󽻣�avxû���룩:";
	gettime(search_list_bit_avx_non, t_query, idx);



cout << "λͼ�洢��ʽ�µİ�Ԫ����:";
	gettime(search_element_bit, t_query, idx);
		cout << "λͼ�洢��ʽ�µİ�Ԫ����sse:";
	gettime(search_element_bit_sse, t_query, idx);
	cout << "λͼ�洢��ʽ�µİ�Ԫ����sse(������):";
	gettime(search_element_bit_sse_non, t_query, idx);
	cout << "λͼ�洢��ʽ�µİ�Ԫ�����󽻣�avx��:";
	gettime(search_element_bit_avx, t_query, idx);

	cout << "λͼ�洢��ʽ�µİ�Ԫ�����󽻣�avxû���룩:";
	gettime(search_element_bit_avx_non, t_query, idx);

	/*cout << "������:";
	gettime(search_list, t_query, idx);

	cout << "��Ԫ����:";
	gettime(search_element, t_query, idx);
*/

	
	return 0;
}