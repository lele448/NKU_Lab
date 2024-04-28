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
class index //索引类
{
public:
	int len = 0;//初始化长度为0
	vector<unsigned int> order;//定义一个无符号整数向量，用于存储索引的顺序

};

bool operator < (const index& s1, const index& s2)
{
	return s1.len < s2.len;// 重载运算符函数，用于比较两个 index 类对象的长度
}
const int row = 100;
const int listnum = 1000;//生成的倒排列表数量


const int querynum =300;


class Bitmap
{
public:
	vector<int> bits0;
	vector<int> bits;//一维数组,bits表示一个倒排列表中包含的doc情况
	vector<int> first_index;
	vector<int> second_index;
	Bitmap(int range)//构造函数 Bitmap(int range)，用于初始化位图，参数 range 表示位图的范围。range 是一个整数，代表了位图中所能包含的最大范围。
	{//位图被分为三部分来存储
	   // this->bits0.resize(range+1);
		this->bits.resize(range / 32 + 1);//将 bits 向量的大小设置为 range / 32 + 1
		//因为每个整数是由 32 个比特位组成的，所以 range / 32 可以得到需要的整数数量，再加 1 是为了确保能够容纳所有可能的整数，因为索引是从 0 开始的。

		this->first_index.resize(range / 1024 + 1);
		this->second_index.resize(range / 32768 + 1);
	}

	void set_value(int data)//设置位图中的某个位置为 1，参数 data 是要设置的位置。
	{
		int index0 = data / 32;//例如data为
		int index1 = index0 / 32;
		int index2 = index1 / 32;

		int tmp0 = data % 32;
		int tmp1 = index0 % 32;
		int tmp2 = index1 % 32;

		//根据位置 data 计算出在 bits、first_index 和 second_index 中的索引。然后，将对应位置的位设置为 1。
		//this->bits0|= (1 << data);
		this->bits[index0] |= (1 << tmp0);
		this->first_index[index1] |= (1 << tmp1);
		this->second_index[index2] |= (1 << tmp2);
	}

	void reset(int data)//将位图中的某个位置重置为 0
	{//首先计算出在 bits 中的索引，并将对应位置的位设置为 0
		int index = data / 32;
		int tmp = data % 32;
		this->bits[index] &= ~(1 << tmp);//设置为0

		//this->bits &= ~(1 << data);
	}

};





index n_index;

vector<index> idx;

Bitmap S(30000000);//S 是 Bitmap 类的对象，用于表示位图，初始化大小为 30000000。


void search_list_bit_sse1(int* query, vector<index>& idx, int num)
{
	vector<index> t_idx;
	for (int i = 0; i < num; i++) {
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());

	// 构建位图数组
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // 初始化位图
		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));  // 移动构造函数
	}

	// 求交集
	S = bitmap[0];  // 移动赋值
	for (int i = 1; i < num; i++) {
		for (size_t j = 0; j < S.second_index.size(); j++) {
			if (S.second_index[j] == 0 || (S.second_index[j] & bitmap[i].second_index[j] == 0)) continue;  // 二级索引为0，直接跳过
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

	// 构建位图数组
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // 初始化位图

		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));  // 移动构造函数
	}

	// 求交集
	Bitmap& S = bitmap[0];

	for (int i = 1; i < num; i++) {
		for (size_t j = 0; j < S.second_index.size(); j += 4) {
			bool judge = false;
			alignas(16) __m128i var0 = _mm_load_si128((__m128i*) & S.second_index[j]);
			alignas(16) __m128i var1 = _mm_load_si128((__m128i*) & bitmap[i].second_index[j]);
			__m128i var = _mm_and_si128(var0, var1);
			if (_mm_movemask_epi8(_mm_cmpeq_epi8(var, _mm_setzero_si128())) != 0xFFFF) {  // 检查 var 是否包含非零元素
				_mm_store_si128((__m128i*) & S.second_index[j], var);
				// 对 first_index 和 bits 进一步处理
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

	// 构建位图数组
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // 初始化位图

		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));  // 移动构造函数
	}

	// 求交集
	Bitmap& S = bitmap[0];

	for (int i = 1; i < num; i++) {
		for (size_t j = 0; j < S.second_index.size(); j += 4) {
			bool judge = false;
			__m128i var0 = _mm_loadu_si128((__m128i*) & S.second_index[j]);
			__m128i var1 = _mm_loadu_si128((__m128i*) & bitmap[i].second_index[j]);
			__m128i var = _mm_and_si128(var0, var1);
			if (_mm_movemask_epi8(_mm_cmpeq_epi8(var, _mm_setzero_si128())) != 0xFFFF) {  // 检查 var 是否包含非零元素
				_mm_store_si128((__m128i*) & S.second_index[j], var);
				// 对 first_index 和 bits 进一步处理
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

	// 构建位图数组
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // 初始化位图

		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));  // 移动构造函数
	}

	// 求交集
	Bitmap& S = bitmap[0];

for (int i = 1; i < num; i++) {
		for (size_t j = 0; j < S.second_index.size()/8; j += 8) {
			bool judge = false;
			alignas(32)__m256i var0 = _mm256_load_si256((__m256i*) & S.second_index[j]);
			alignas(32)__m256i var1 = _mm256_load_si256((__m256i*) & bitmap[i].second_index[j]);
			__m256i var = _mm256_and_si256(var0, var1);
			if (!_mm256_testz_si256(var, var)) {  // 检查 var 是否包含非零元素
				_mm256_store_si256((__m256i*) & S.second_index[j], var);


				// 对 first_index 和 bits 进一步处理
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

	// 构建位图数组
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // 初始化位图

		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));  // 移动构造函数
	}

	// 求交集
	Bitmap& S = bitmap[0];

	for (int i = 1; i < num; i++) {
		for (size_t j = 0; j < S.second_index.size() / 8; j += 8) {
			bool judge = false;
			__m256i var0 = _mm256_loadu_si256((__m256i*) & S.second_index[j]);
			__m256i var1 = _mm256_loadu_si256((__m256i*) & bitmap[i].second_index[j]);
			__m256i var = _mm256_and_si256(var0, var1);
			if (!_mm256_testz_si256(var, var)) {  // 检查 var 是否包含非零元素
				_mm256_store_si256((__m256i*) & S.second_index[j], var);


				// 对 first_index 和 bits 进一步处理
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

//二级
void search_list_bit(int* query, vector<index>& idx, int num)
{
	vector<index> t_idx;
	for (int i = 0; i < num; i++) {
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());

	// 构建位图数组
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++) {
		Bitmap bm(30000000);  // 初始化位图
		for (int j = 0; j < t_idx[i].len; j++) {
			bm.set_value(t_idx[i].order[j]);
		}
		bitmap.push_back(move(bm));  // 移动构造函数
	}

	// 求交集
	S = bitmap[0];  // 移动赋值
	for (int i = 1; i < num; i++) {
		for (size_t j = 0; j < S.second_index.size(); j++) {
			if (S.second_index[j] == 0 || (S.second_index[j] & bitmap[i].second_index[j] == 0)) continue;  // 二级索引为0，直接跳过
			for (size_t t = j * 32; t < (j + 1) * 32; t++) {
				if (S.first_index[t] == 0 || (S.first_index[t] & bitmap[i].first_index[t] == 0)) continue;  // 一级索引为0，直接跳过
				for (size_t k = t * 32; k < (t + 1) * 32; k++) {
					if (S.bits[k] == 0) continue;  // 位为0，直接跳过
					S.bits[k] &= bitmap[i].bits[k];  // 按位与操作
				}
			}
		}
	}
}



//一个一个元素：(32个元素为一个数组元素)
void search_list_bit0(int* query, vector<index>& idx, int num)//用于在使用位图表示的索引列表中进行搜索
{
	vector<index> t_idx;//创建索引类的向量，用于存储根据查询位置从 idx 中取得的索引信息
	for (int i = 0; i < num; i++)
	{
		t_idx.push_back(idx[query[i]]);//t_idx存放所有倒排列表的数据
	}
	sort(t_idx.begin(), t_idx.end());//排序
	vector<Bitmap> bitmap;
	for (int i = 0; i < num; i++)
	{
		bitmap.push_back(30000000);//初始化
		for (int j = 0; j < t_idx[i].len; j++)//j为一个t_idx[]倒排列表的长度
		{
			bitmap[i].set_value(t_idx[i].order[j]);//放入bitmap
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
				if (S.bits[j] != 0)//不等于0，说明为1，则相同
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

//此函数的主要功能是根据查询的索引位置，使用位图表示的索引列表进行搜索，通过位运算对位图进行更新，最终得到匹配的结果位图 S。
void search_element_bit(int* query, vector<index>& idx, int num)//在使用位图表示的索引列表中搜索共同的元素
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
		Bitmap bm(30000000);  // 假设存在这样一个构造函数初始化大小
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
			if (mask != 0xFFFF) { // 如果不全为0
				var0 = var; // 更新 S.second_index 直接在寄存器中进行
				for (int t = i * 32; t < i * 32 + 32; t += 4) {
					alignas(16)__m128i a0 = _mm_load_si128((__m128i*) & S.first_index[t]);
					alignas(16)__m128i a1 = _mm_load_si128((__m128i*) & bitmap[j].first_index[t]);
					__m128i a = _mm_and_si128(a0, a1);
					int first_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(a, _mm_setzero_si128()));
					if (first_mask != 0xFFFF) {
						a0 = a; // 更新 S.first_index
						for (int l = t * 32; l < t * 32 + 32; l += 4) {
							alignas(16)__m128i b0 = _mm_load_si128((__m128i*) & S.bits[l]);
							alignas(16)__m128i b1 = _mm_load_si128((__m128i*) & bitmap[j].bits[l]);
							__m128i b = _mm_and_si128(b0, b1);
							int bits_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(b, _mm_setzero_si128()));
							if (bits_mask != 0xFFFF) {
								judge = true; // 更新 S.bits
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
		Bitmap bm(30000000);  // 假设存在这样一个构造函数初始化大小
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
			if (mask != 0xFFFF) { // 如果不全为0
				var0 = var; // 更新 S.second_index 直接在寄存器中进行
				for (int t = i * 32; t < i * 32 + 32; t += 4) {
					__m128i a0 = _mm_loadu_si128((__m128i*) & S.first_index[t]);
					__m128i a1 = _mm_loadu_si128((__m128i*) & bitmap[j].first_index[t]);
					__m128i a = _mm_and_si128(a0, a1);
					int first_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(a, _mm_setzero_si128()));
					if (first_mask != 0xFFFF) {
						a0 = a; // 更新 S.first_index
						for (int l = t * 32; l < t * 32 + 32; l += 4) {
							__m128i b0 = _mm_loadu_si128((__m128i*) & S.bits[l]);
							__m128i b1 = _mm_loadu_si128((__m128i*) & bitmap[j].bits[l]);
							__m128i b = _mm_and_si128(b0, b1);
							int bits_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(b, _mm_setzero_si128()));
							if (bits_mask != 0xFFFF) {
								judge = true; // 更新 S.bits
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
		Bitmap bm(30000000);  // 假设存在这样一个构造函数初始化大小
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

			if (!_mm256_testz_si256(var, var)) { // 如果不全为0
				var0 = var; // 更新 S.second_index 直接在寄存器中进行
				for (int t = i * 32; t < (i * 32 + 32)/8; t += 8) {
					alignas(32)__m256i a0 = _mm256_load_si256((__m256i*) & S.first_index[t]);
					alignas(32)__m256i a1 = _mm256_load_si256((__m256i*) & bitmap[j].first_index[t]);

					__m256i  a = _mm256_and_si256(a0, a1);
					if (!_mm256_testz_si256(a, a)) {
						a0 = a; // 更新 S.first_index
						for (int l = t * 32; l < (t * 32 + 32)/8; l += 8) {
							alignas(32)__m256i b0 = _mm256_load_si256((__m256i*) & S.bits[l]);
							alignas(32)__m256i b1 = _mm256_load_si256((__m256i*) & bitmap[j].bits[l]);
							__m256i b = _mm256_and_si256(b0, b1);
							
							if (!_mm256_testz_si256(b, b)) {
								judge = true; // 更新 S.bits
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
		Bitmap bm(30000000);  // 假设存在这样一个构造函数初始化大小
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

			if (!_mm256_testz_si256(var, var)) { // 如果不全为0
				var0 = var; // 更新 S.second_index 直接在寄存器中进行
				for (int t = i * 32; t < (i * 32 + 32) / 8; t += 8) {
					__m256i a0 = _mm256_loadu_si256((__m256i*) & S.first_index[t]);
					__m256i a1 = _mm256_loadu_si256((__m256i*) & bitmap[j].first_index[t]);

					__m256i  a = _mm256_and_si256(a0, a1);
					if (!_mm256_testz_si256(a, a)) {
						a0 = a; // 更新 S.first_index
						for (int l = t * 32; l < (t * 32 + 32) / 8; l += 8) {
							__m256i b0 = _mm256_loadu_si256((__m256i*) & S.bits[l]);
							__m256i b1 = _mm256_loadu_si256((__m256i*) & bitmap[j].bits[l]);
							__m256i b = _mm256_and_si256(b0, b1);

							if (!_mm256_testz_si256(b, b)) {
								judge = true; // 更新 S.bits
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

void search_element_bit0(int* query, vector<index>& idx, int num)//在使用位图表示的索引列表中搜索共同的元素
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
		for (int j = 1; j < num; j++)//列表数量
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

	// 存储当前交集
	vector<unsigned int> intersection = t_idx[0].order;

	for (int i = 1; i < num; i++) {
		// 临时存储新的交集
		vector<unsigned int> new_intersection;

		// 使用二分查找在当前链表中查找是否存在于交集中的元素
		for (unsigned int elem : intersection) {
			if (binary_search(t_idx[i].order.begin(), t_idx[i].order.end(), elem)) {
				new_intersection.push_back(elem);
			}
		}

		// 更新当前交集为新的交集
		intersection = new_intersection;
	}

	// 更新当前交集的长度
	n_index.len = intersection.size();
	// 更新当前交集的顺序
	n_index.order = move(intersection);
}


void search_element(int* query, vector<index>& idx, int num) {
	vector<index> t_idx;
	for (int i = 0; i < num; i++) {
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());

	// 选择最短链表
	index& shortest_list = t_idx[0];
	for (int i = 0; i < shortest_list.len; i++) {
		unsigned int elem = shortest_list.order[i];
		bool found_in_all = true;

		// 遍历其他链表进行二分查找
		for (int j = 1; j < num; j++) {
			index& current_list = t_idx[j];
			if (!binary_search(current_list.order.begin(), current_list.order.end(), elem)) {
				found_in_all = false;
				break;
			}
		}

		// 如果元素存在于所有链表中，加入交集
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
	for (int i = 0; i < 2000; i++)//2000个文档
	{
		index tmp;//每次迭代都从文件中读取一个 index 对象的数据,tmp是一个倒排列表
		outfile.read((char*)&tmp.len, sizeof(tmp.len));//首先读取 len 成员变量的值，然后根据 len 的大小读取对应数量的 order 数据，并将它们存储到 tmp 中
		for (int j = 0; j < (tmp.len); j++)
		{//n_tmp为doc文档
			unsigned int n_tmp;//暂存从文件中读取的数据
			outfile.read((char*)&n_tmp, sizeof(n_tmp));//从文件中读取一个四字节无符号整数（小端），并将其存储到 n_tmp 中
			tmp.order.push_back(n_tmp);//将读取的无符号整数数据 n_tmp 添加到 tmp 对象的 order 向量中
		}
		idx.push_back(tmp);//将读取的 tmp 对象添加到名为 idx 的 index 向量中
		//tmp 用于暂存一组数据以及其长度信息，而 idx 则是存储了多个 tmp 对象的容器。
	}

	outfile.close();//关闭索引文件流 outfile

	outfile.open("ExpQuery", ios::in);
	//1000次查询记录，每次查询最多查5个倒排列表
	int t_query[querynum][5] = { 0 };//声明一个二维整数数组 t_query[1000][5] 并初始化为全零
	string line;
	int n_count = 0;

	//以下函数就是将ExpQuery文件中的数据转换为数组存储

	while (getline(outfile, line) && n_count < querynum)//使用一个循环，读取文件的每一行，每行代表一个查询
	{
		stringstream ss(line);//对于每一行，使用 stringstream 对象 ss 进行分词，将其中的整数提取出来，并存储到 t_query 数组中。
		int addr = 0;
		while (!ss.eof())
		{
			int tmp;
			ss >> tmp;
			t_query[n_count][addr] = tmp;//计数器 n_count 用于记录读取的查询行数
			addr++;
		}
		n_count++;
	}
	//t_query数组存的是查询记录，而idx存的是倒排链表，存储了多个 tmp 对象的容器
	outfile.close();
	/*
//
	srand(time(0));
	for (int i = 0; i < listnum; i++)
	{
		index t;// 生成500个倒排列表
		t.len= rand() % (150 - 50 + 1) + 30;//30~80
		vector<int> forRandom;//整数型数组
		for (int j = 0; j < t.len * 4; j++)//为了不重复
		{
			forRandom.push_back(j);
		}
		random_shuffle(forRandom.begin(), forRandom.end());//随机打乱

		for (int j = 0; j < t.len; j++)
		{
			int docId = forRandom[j];

			t.order.push_back(docId);
		}
		sort(t.order.begin(), t.order.end());//对文档编号排序

		idx.push_back(t);

	}



int t_query[row][5]={ 0 };
	for (int i = 0; i < row; i++)
		{// 做200次查询
		//int testQuery[5];
		for (int j = 0; j < 5; j++)
				{//cout<< j << endl;
			t_query[i][j] = rand() % listnum;
	   }
	}*/
	//
	cout << "querynum=" << querynum<<endl;

	cout << "位图存储方式下的按表求求交（二级索引）:";
	gettime(search_list_bit, t_query, idx);

	cout << "位图存储方式下的按表求求交（sse1）:";
	gettime(search_list_bit_sse1, t_query, idx);
	
	cout << "位图存储方式下的按表求求交（sse）:";
	gettime(search_list_bit_sse, t_query, idx);
	cout << "位图存储方式下的按表求求交（sse不对齐）:";
	gettime(search_list_bit_sse_non, t_query, idx);
	cout << "位图存储方式下的按表求求交（avx）:";
	gettime(search_list_bit_avx, t_query, idx);
	cout << "位图存储方式下的按表求求交（avx没对齐）:";
	gettime(search_list_bit_avx_non, t_query, idx);



cout << "位图存储方式下的按元素求交:";
	gettime(search_element_bit, t_query, idx);
		cout << "位图存储方式下的按元素求交sse:";
	gettime(search_element_bit_sse, t_query, idx);
	cout << "位图存储方式下的按元素求交sse(不对齐):";
	gettime(search_element_bit_sse_non, t_query, idx);
	cout << "位图存储方式下的按元素求求交（avx）:";
	gettime(search_element_bit_avx, t_query, idx);

	cout << "位图存储方式下的按元素求求交（avx没对齐）:";
	gettime(search_element_bit_avx_non, t_query, idx);

	/*cout << "按表求交:";
	gettime(search_list, t_query, idx);

	cout << "按元素求交:";
	gettime(search_element, t_query, idx);
*/

	
	return 0;
}