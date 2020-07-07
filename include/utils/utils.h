#pragma once
#include "rsearch_type.h"
#include <bits/stdc++.h>
#include <sys/sysinfo.h>
#include <cxxabi.h>
namespace rsearch{
using std::pair;
using std::make_pair;
using std::vector;
using std::ifstream;
using std::ofstream;

// ==================== public functions ==================
// this function may be used while you use the library
inline pair<float, float> __float_7bits(float* data, int64_t nt){
    if (nt < 100)
        return make_pair(-1, 0);
    int64_t n = std::min(nt, (int64_t)200000);
    std::vector<float> tmp(n);
    memcpy(tmp.data(), data, n * sizeof(float));
    //std::cout << "__float_7bits: "  << tmp[0] << std::endl;
    int64_t l = 0.1 * n;
    int64_t r = 0.9 * n;
    float max_v, min_v;
    std::nth_element(tmp.data(), tmp.data() + l, tmp.data() + n);
    min_v = tmp[l];
    std::nth_element(tmp.data(), tmp.data() + r, tmp.data() + n);
    max_v = tmp[r];
    float k = 126 / (max_v - min_v);
    //float b = 70 - k * max_v;
    float b = 63 - k * max_v;
    std::cout << "__float_7bits: "  << n << " " << max_v << " " << min_v  << " " << k  << " " << b << std::endl;
    return make_pair(k, b);
}

inline float float_7bits(const float* data, int8_t* td, int64_t n, float k = 463.0, float b = 0){
    for (int64_t i = 0; i < n; ++i){
        td[i] = std::max((int)-63, std::min(63, (int)(k * data[i] + b)));
    }
    return 0;
}



// ==================== pair compare ==================== 
template<typename T1, typename T2>
struct pair_greator{
    bool operator()(pair<T1, T2> const &a, pair<T1, T2> const &b) const{
        if (a.first == b.first) {
            return a.second < b.second;
        }
        return a.first > b.first; 
    }
};

template<typename T1, typename T2>
struct is_same_type
{
    operator bool()
    {
        return false;
    }
};
 
template<typename T1>
struct is_same_type<T1, T1>
{
    operator bool()
    {
        return true;
    }
};

// ==================== math ==================== 
template<typename T>
inline typemap_t<T> dot_prod(const T* d1, const T* d2, int dimension){
    using Tout = typemap_t<T>;
    Tout ans = 0;
    for (int i = 0; i < dimension; ++i){
        ans += (Tout)d1[i] * (Tout)d2[i];
    }
    return ans;
}

template<typename T>
inline typemap_t<T> sims_two(const T* d1, const T* d2, int dimension){
    using Tout = typemap_t<T>;
    Tout ans = 0;
    for (int i = 0; i < dimension; ++i){
        ans += ((Tout)d1[i] - (Tout)d2[i]) * ((Tout)d1[i] - (Tout)d2[i]);
    }
    return ans;
}

template <typename T,
          DistanceType dist_type>
inline typemap_t<T> vec_dis(const T* d1, const T* d2, int dimension){
    return sims_two(d1, d2, dimension);
}

template <>
inline int vec_dis<int8_t, EUCLIDEAN>(const int8_t* d1, const int8_t* d2, int dimension){
    return sims_two(d1, d2, dimension);
}

template <>
inline int vec_dis<int8_t, COSINE>(const int8_t* d1, const int8_t* d2, int dimension){
    return dot_prod(d1, d2, dimension);
}

template <>
inline float vec_dis<float, EUCLIDEAN>(const float* d1, const float* d2, int dimension){
    return sims_two(d1, d2, dimension);
}

template <>
inline float vec_dis<float, COSINE>(const float* d1, const float* d2, int dimension){
    return dot_prod(d1, d2, dimension);
}

template<typename T,
        DistanceType dist_type>
inline typemap_t<T> get_offset(const T* d, int dimension){
    typemap_t<T> v = 0;
    if (dist_type == EUCLIDEAN)
        v -= dot_prod<T>(d, d, dimension) / 2;
    if (is_same_type<T, int8_t>() == true)
        for (int i = 0; i < dimension; ++i)
            v -= (int)64 * d[i];
    return v;
}

inline void norm(float* data, int n, int dimension, float scale = 1.0){
    for (int i = 0 ; i < n ; ++i){
        float len = sqrt(dot_prod<float>(data + 1LL * i * dimension, data + 1LL * i * dimension, dimension));
        //if (i == 1000) std::cout << "[norm] len = %d" << len << std::endl;
        for (int j = 0; j < dimension; ++j) data[1LL * i * dimension + j] /= (len / scale);
    }
}

template<typename T>
void divide(const T* src, T* dst, int n, int dimension, int div_dimension){
    int len = dimension / div_dimension;
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < len; ++j)
            memcpy(dst + 1LL * j * (n * div_dimension) + 1LL * i * div_dimension, src + 1LL * i * dimension + 1LL * j * div_dimension, sizeof(T) * div_dimension);
    }
}

template<typename T>
void merge(pair<T,  idx_t>* cache, pair<T,  idx_t>* a, pair<T,  idx_t>* b, int a_sz, int b_sz){
    int i=0, j=0, k=0;
    
    while (true){
        if (k >= a_sz || j >= b_sz) break;
        if (a[i].second != -1 && a[i].first > b[j].first){
            cache[k] = a[i];
            ++i;
            ++k;
        }
        else {
            cache[k] = b[j];
            ++j;
            ++k;
        }
    }
    if (k < a_sz)
        memcpy(cache + k, a + i, (a_sz - k) * sizeof(pair<T, idx_t>));
    memcpy(a, cache, a_sz * sizeof(pair<T, idx_t>));
}
/*
template<typename T1,
        typename T2,
        DistanceType dist_type>
inline void get_code_v1(const T1* x, const T1* pq, const typemap_t<T1>* pq_offset, int n, int pq_num, int pq_dimension, int dimension, T2* code){
    using Tout = typemap_t<T1>;
    Tout d, min_dis;
    r_dot_prod<T1>(x, pq, pq_offset, n * code_len, pq_num, pq_dimension, , pq_num);
    for (int i = 0; i < n; ++i){
        
    }
    for (int j = 0, _j=0; j < dimension; j += pq_dimension, ++_j){
        min_idx = 0;
        min_dis = vec_dis<T1, dist_type>(x + j, pq + j, pq_dimension);
        for (int k = 1; k < pq_num; ++k){
            d = vec_dis<T1, dist_type>(x + 1LL * i * dimension + j, pq + 1LL * k * pq_dimension, pq_dimension);
            if (d < min_dis){
                min_dis = d;
                min_idx = k;
            }
            code[i][_j] = min_idx;
        }
    }
}*/

// ==================== I/O ==================== 
template<typename T>
inline void r_read(ifstream &fin, T* x, int64_t n){
    fin.read((char*)x, n * sizeof(T));
}

template<typename T>
inline void r_write(ofstream &fout, T* x, int64_t n){
    fout.write((char*)x, n * sizeof(T));
}

template<typename T>
inline void r_file2bytes(ifstream &fin, T* x, int& n, int& dimension){
    r_read<int32_t>(fin, &n, 1);
    r_read<int32_t>(fin, &dimension, 1);
    r_read<T>(fin, x, 1LL * n * dimension);   
}

template<typename T>
inline void r_file2bytes(ifstream &fin, vector<T>& x, int& n, int& dimension){
    r_read<int32_t>(fin, &n, 1);
    r_read<int32_t>(fin, &dimension, 1);
    std::cout << n << " " << dimension << std::endl;
    x.resize(1LL * n * dimension);
    r_read<T>(fin, x.data(), 1LL * n * dimension);   
}

template<typename T>
inline void r_bytes2file(ofstream &fout, T* x, int &n, int &dimension){
    r_write<int32_t>(fout, &n, 1);
    r_write<int32_t>(fout, &dimension, 1);
    r_write<T>(fout, x, 1LL * n * dimension);
}

inline bool file_exist(const char *file_name)
{
    std::ifstream infile(file_name);
    return infile.good();
};

inline bool file_exist(std::string file_name){
    std::ifstream infile(file_name);
    return infile.good();
}

// ==================== Create data ==================== 

inline int init_random(float* data, int64_t n){
    const int MO = 65535;
    for (int64_t i = 0; i < n; ++i)
        data[i] = 2.0 * (rand() % MO) / MO - 1.0;
    return 0;
}

void __get_random_data(float* data, int n, int dimension);

template<typename T,
        DistanceType dist_type>
void get_random_data(vector<T>& data, int n, int dimension);

}
