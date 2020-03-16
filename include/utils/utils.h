#pragma once
#include <bits/stdc++.h>
#include <sys/sysinfo.h>
#include <cxxabi.h>
#include "rsearch_type.h"
namespace rsearch{
using std::pair;
using std::vector;
using std::ifstream;
using std::ofstream;
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
    if (dist_type == EUCLIDEAN){
        return -dot_prod<T>(d, d, dimension) / 2;
    }
    else{
        return 0;
    }
}
/*
template<typename T1,
        typename T2,
        DistanceType dist_type>
inline float get_code_v1(const T1* x, const T1* pq, int n, int pq_num, int pq_dimension, int dimension, T2* code){
    using Tout = typemap_t<T1>;
    for (int i = 0; i < n; ++i)
    for (int j = 0, _j=0; j < dimension; j += pq_dimension, ++_j){
        int min_idx = 0;
        Tout min_dis = vector_dis<dist_type>(x + j, pq + j, pq_dimension);
        for (int k = 1; k < n; ++k)
            Tout dis = vec_dis<dist_type>(x + 1LL * i * dimension + j, y + 1LL * k * dimension + j, pq_dimension);
            if (dis < min_dis){
                min_dis = dis;
                min_idx = k;
            }
            code[i][_j] = min_idx;
    }

}
*/

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

// ==================== I/O ==================== 
template<typename T>
inline void r_read(ifstream &fin, T* x, int n){
    fin.read((char*)x, n * sizeof(T));
}

template<typename T>
inline void r_write(ofstream &fout, T* x, int n){
    fout.write((char*)x, n * sizeof(T));
}

template<typename T>
inline void r_file2bytes(ifstream &fin, T* x, int& n, int& dimension){
    r_read<int32_t>(fin, &n, 1);
    r_read<int32_t>(fin, &dimension, 1);
    r_read<T>(fin, x, n * dimension);   
}

template<typename T>
inline void r_file2bytes(ifstream &fin, vector<T>& x, int& n, int& dimension){
    r_read<int32_t>(fin, &n, 1);
    r_read<int32_t>(fin, &dimension, 1);
    x.reserve(1LL * n * dimension);
    r_read<T>(fin, x.data(), n * dimension);   
}

template<typename T>
inline void r_bytes2file(ofstream &fout, T* x, int &n, int &dimension){
    r_write<int32_t>(fout, &n, 1);
    r_write<int32_t>(fout, &dimension, 1);
    r_write<T>(fout, x, n * dimension);
}


inline void norm(float* data, int n, int dimension){
    for (int i = 0 ; i < n ; ++i){
        float len = sqrt(dot_prod<float>(data + 1LL * i * dimension, data + 1LL * i * dimension, dimension));
        for (int j = 0; j < dimension; ++j) data[1LL * i * dimension + j] /= len;
    }
}

inline float float_7bits(float* data, int8_t* td, int64_t n){
    float max_v, min_v;
    for (int64_t i = 0; i < n; ++i){
        max_v = std::max(max_v, data[i]);
        min_v = std::min(min_v, data[i]);
    }
    
    for (int64_t i = 0; i < n; ++i){
        max_v = std::max(max_v, data[i]);
        min_v = std::min(min_v, data[i]);
    }
    float k = 140 / (max_v - min_v);
    //float b = 70 - k * max_v;
    for (int64_t i = 0 ;i < n; ++i){
        td[i] = std::max((int)-63, std::min(63, (int)(k * data[i])));
    }
    return k;
}

inline float float_7bits(const float* data, int8_t* td, int64_t n, float k){
    for (int64_t i = 0 ;i < n; ++i){
        td[i] = std::max((int)-63, std::min(63, (int)(k * data[i])));
    }
    return 0;
}

inline bool file_exist(const char *file_name)
{
    std::ifstream infile(file_name);
    return infile.good();
};

template<typename T = float>
int init_random(T* data, int n, int dimension){
    const int MO = 65535;
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < dimension; ++j){
            data[i * dimension + j] = 1.0 * (rand() % MO) / MO;
        }
    }
    return 0;
}

template<>
int init_random<int8_t>(int8_t* data, int n, int dimension){
    const int MO = 65535;
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < dimension; ++j){
            data[i * dimension + j] = 1.0 * (rand() % MO) / MO * 453.0;
        }
    }
    return 0;
}

template<typename T>
void get_random_data(T* data, int n, int dimension){
    ofstream fout;
    int status;
    char *type_name = abi::__cxa_demangle(typeid(T).name(), NULL,  NULL, &status);
    char fname[200];
    sprintf(fname, "/home/zzr/data/.rsearch.%s.%d.%d.bin", type_name, dimension, n);
    
    if (file_exist(fname)){
        ifstream fin(fname, ifstream::binary);
        r_file2bytes<T>(fin, data, n, dimension);
    } else {
        fout.open(fname, ofstream::binary);
        init_random(data, dimension, n);
        r_bytes2file<T>(fout, data, n, dimension);
    }
}


}
