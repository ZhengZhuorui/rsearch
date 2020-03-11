#pragma once
#include <bits/stdc++.h>

namespace rsearch{

// ---------------ERROR NO. -------------

enum Errorno{
    INDEX_EXISTS = -1,
    INDEX_NO_FIND = -2,
    TRAINDATA_ERROR = -3,
    NOT_TRAIN = -4,
};


using uint64_t = unsigned long long;
using uint32_t = unsigned int;
using int64_t = long long;
using int32_t = int;
using int8_t = signed char;
using uint8_t = unsigned char;
using idx_t = int32_t;
typedef enum DistanceType{
    COSINE = 0,
    EUCLIDEAN = 1,
} DistanceType;

typedef enum MethodType{
    DUMMY = 0,
    X86_RAPID = 1, 
    X86_MULTI_THREAD = 2,
    X86_MULTI_PLATFORM = 3,
    X86_FAISS_FLAT = 4,

    
} MethodType;

template<typename T>
struct TMap{};

template <>
struct TMap<int8_t>{
    typedef int type;
};

template <>
struct TMap<float>{
    typedef float type;
};

template<typename T>
using typemap_t = typename TMap<T>::type;

}