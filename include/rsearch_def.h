#pragma once
#include "rsearch_type.h"
namespace rsearch{

//probe:
template<typename T> class probe;
template<typename T, DistanceType dist_type, typename matrix_type> class base_probe;
template<typename T, DistanceType dist_type, typename matrix_type> class cpu_base_probe;
template<typename T, DistanceType dist_type, typename matrix_type> class cpu_base_mt_probe;
template<typename T, DistanceType dist_type> class pqivf_probe;
template<typename T, DistanceType dist_type> class pqivf_mt_probe;
template<typename T> class faiss_probe;


//gallery
template<typename T> class gallery;
template<typename T, DistanceType dist_type> class cpu_base_gallery;
template<typename T, DistanceType dist_type> class pqivf_gallery;
template<typename T> class faiss_gallery;
template<typename T> class simple_gallery;

//matrix_mul
template<typename T> class matrix_mul;
template<typename T> class base_matrix_mul;
template<typename T> class rapid_matrix_mul;
template<typename T> class matrix_la;
template<typename T> class rapid_matrix_la;

template<typename T> class simple_index;
//function
//template<typename T> probe<T>* create_probe(int dimension, int topk, MethodType method_type);

}