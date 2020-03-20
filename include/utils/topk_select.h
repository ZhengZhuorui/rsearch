#include <bits/stdc++.h>
namespace rsearch{
template<typename T>
void min_heapify(T *key, int *index_out, const int start, const int end);
template<typename T>
void max_heapify(T *key, int *index_out, const int start, const int end);
template<typename T, bool DESCEND>
void top1(const T *key, T *key_out, int* index_out, const int key_len);
template<typename T, bool DESCEND>
void topKIndex(const T *key, T *key_out, int* index_out, const int key_len, const int K);
template<typename T>
int cpu_select_kv(const T * key, T *key_out, int * index_out, int topk, int size, int batch_size, int ldc, bool DESCEND);
}