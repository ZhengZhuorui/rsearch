#pragma once
#include "rsearch_type.h"
namespace rsearch{

//probe:
template<typename T, DistanceType dist_type> class probe;
template<typename T, DistanceType dist_type, typename matrix_type> class base_probe;
template<typename T, DistanceType dist_type, typename matrix_type> class cpu_base_probe;
template<typename T, DistanceType dist_type> class pqivf_probe;

//gallery
template<typename T, DistanceType dist_type> class gallery;
template<typename T, DistanceType dist_type> class cpu_base_gallery;
template<typename T, DistanceType dist_type> class pqivf_gallery;

//matrix_mul
template<typename T> class matrix_mul;
template<typename T> class base_matrix_mul;
template<typename T> class rapid_matrix_mul;

}