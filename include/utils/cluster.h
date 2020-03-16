#pragma once
#include "rsearch_type.h"
#include <bits/stdc++.h>
#include "utils/utils.h"
#include "matrix/rapid_matrix_mul.h"
namespace rsearch{
using std::pair;
template<typename T,
        DistanceType dist_type>
int k_means(const T* data, int n, int cluster_center, int dimension, std::vector<T>& res){
    using Tout = typemap_t<T>;
    //srand(time(NULL));
    res.reserve(cluster_center * dimension);
    vector<int> cluster_num(cluster_center);
    vector<Tout> res_offset(cluster_center);
    vector<Tout> cluster_v(cluster_center * dimension);
    vector<T> res_tmp(cluster_center * dimension);
    memset(res_tmp.data(), 0, cluster_center * dimension * sizeof(T));
    memset(cluster_v.data(), 0, cluster_center * dimension * sizeof(Tout));
    memest(cluster_num.data(), 0, cluster_center * sizeof(int));
    for (int i = 0; i < cluster_center; ++i){
        int id = rand() % n;
        memcpy(res.data() + i * dimension, data + id * dimension, this->dimension);
        res_offset = get_offset<T, dist_type>(res.data() + i * dimension, this->dimension);
    }
    matrix_mul<T>* mm;
    pair<T, idx_t> dis;
    mm = new rapid_matrix_mul<T>;
    int max_batch = 64;
    mm->set(dimension, 1, max_batch, cluster_center);
    int i;
    for (i = 0; i < 15; ++i){
        int flag;
        for (int j = 0; j < n; j += max_batch){
            int size = std::min(max_batch, cluster_center);
            mm->mul(data + j * dimension, res.data(), res_offset.data(), size, dis);
            for (int k = 0; k < size; ++k){
                cluster_num[dis[k].second]++;
                for (int l = 0; l < dimension; ++l){
                    cluster_v[dis[k].second * dimension + l] += data[(j + k) * dimension + l];
                }
            }
        }
        memcpy(res_tmp.data(), res.data(), cluster_center * dimension * sizeof(T));
        for (int j = 0; j < cluster_center; ++j){
            if (cluster_num[j] == 0){
                int id = rand() % n;
                memcpy(res.data() + i * dimension, data + id * dimension, this->dimension);
            }
            else{
                for (int k = 0; k < dimension; ++k){
                    res[j * dimension + k] = cluster_v[j * this->dimension + k] / cluster_num[j];
                }
            }
        }
        for (int j = 0; j < cluster_center * dimension; ++j) flag += (res[j] != res_tmp[j]);
        if (flag == 0) break;
    }
    printf("[kmeans]train cycle: %d\n", i);
    return 0;
}
}