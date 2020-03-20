#include "utils/topk_select.h"
namespace rsearch{

template<typename T>
void min_heapify(T *key, int *index_out, const int start, const int end) {
    int dad = start;
    int son = dad * 2 + 1;
    while (son <= end) { 
        //smaller one
        if (son + 1 <= end && key[son] > key[son + 1]) 
            son++;
        //satisfy
        if (key[dad] <= key[son]) 
            return;
        else { 
            std::swap(key[dad], key[son]);
            std::swap(index_out[dad], index_out[son]);
            dad = son;
            son = dad * 2 + 1;
        }
    }
}

template<typename T>
void max_heapify(T *key, int *index_out, const int start, const int end) {
    int dad = start;
    int son = dad * 2 + 1;
    while (son <= end) { 
        //smaller one
        if (son + 1 <= end && key[son] < key[son + 1]) 
            son++;
        //satisfy
        if (key[dad] >= key[son]) 
            return;
        else { 
            std::swap(key[dad], key[son]);
            std::swap(index_out[dad], index_out[son]);
            dad = son;
            son = dad * 2 + 1;
        }
    }
}


template<typename T, bool DESCEND>
void top1(const T *key, T *key_out, int* index_out, const int key_len) {
    T top_val = key[0];
    int top_id = 0;
    for(int i = 1; i < key_len; ++i){
        if(DESCEND){
            if(key[i] > top_val){
                top_val = key[i];
                top_id = i;
            }
        }else{
            if(key[i] < top_val){
                top_val = key[i];
                top_id = i;
            }
        }
    }
    key_out[0] = top_val;
    index_out[0] = top_id;
}



template<typename T, bool DESCEND>
void topKIndex(const T *key, T *key_out, int* index_out, const int key_len, const int K) {

    if(key_len == 1) {
        key_out[0] = key[0];
        index_out[0] = 0;
        return;
    }

    if(K==1) {
        return top1<T, DESCEND>(key, key_out, index_out, key_len);
    }

    //to do, optimize full sort
    // if(key_len <= K){
    // }

    for(int i = 0; i < K; ++i){
        key_out[i] = key[i];
        index_out[i] = i;
    }
    if(DESCEND){
        for (int i = K / 2 - 1; i >= 0; --i)
            min_heapify(key_out, index_out, i, K - 1);
        for(int i = K; i<key_len; ++i){
            if(key[i] > key_out[0]){
                key_out[0] = key[i];
                index_out[0]  = i;
                min_heapify(key_out, index_out, 0, K - 1);
            }
        }
    }
    else {
        for (int i = K / 2 - 1; i >= 0; --i)
            max_heapify(key_out, index_out, i, K - 1);
        for(int i = K; i<key_len; ++i){
            if(key[i] < key_out[0]){
                key_out[0] = key[i];
                index_out[0]  = i;
                max_heapify(key_out, index_out, 0, K - 1);
            }
        }
    }

    for (int i = K - 1; i > 0; --i) {
        std::swap(key_out[0], key_out[i]);
        std::swap(index_out[0],  index_out[i]);
        if(DESCEND)
            min_heapify(key_out, index_out, 0, i - 1);
        else
            max_heapify(key_out, index_out, 0, i - 1);
    }
}
 


template<typename T>
int cpu_select_kv(const T * key, T *key_out, int * index_out, int topk, int size, int batch_size, int ldc, bool DESCEND){
    if(!key || !key_out || !index_out)
        return -1;
    if(size < 1 || topk < 1 || batch_size < 1)
        return -1;
    if(topk > size) 
        topk = size;
    if (DESCEND == true){
        for(int i=0; i < batch_size; ++i)
            topKIndex<T, true>(key + i * ldc, key_out + i * topk, index_out + i * topk, size, topk);
    }
    else{
        for(int i=0; i < batch_size; ++i)
            topKIndex<T, false>(key + i * ldc, key_out + i * topk, index_out + i * topk, size, topk);
    }
    return 0;
}
template int cpu_select_kv<int>(const int * key, int *key_out, int * index_out, int topk, int size, int batch_size, int ldc, bool DESCEND);
template int cpu_select_kv<float>(const float * key, float *key_out, int * index_out, int topk, int size, int batch_size, int ldc, bool DESCEND);

}
