#pragma once
#include "rsearch_def.h"
#include <bits/stdc++.h>
#include "utils/utils.h"
#include "matrix/rapid_matrix_mul.h"
#define DEBUG_KMEANS

namespace rsearch{
using std::pair;
template<typename T,
        DistanceType dist_type>
int k_means(const T* data, const int n, const int cluster_center, const int dimension, std::vector<typemap_t<T> >& res);
}