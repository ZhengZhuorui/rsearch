#include "utils/cluster.h"
namespace rsearch{
using std::pair;
template<typename T,
        DistanceType dist_type>
int k_means(const T* data, const int n, const int cluster_center, const int dimension, std::vector<T>& res){
    //using Tout = typemap_t<T>;
    #ifdef DEBUG_KMEANS
    std::cout << "[k_means] begin, cluster = " << cluster_center << "dimension = "<< dimension << std::endl;
    //for (int i = 0; i < dimension; ++i)
    //    std::cout << (float)data[i] << "?";
    //std::cout << std::endl;
    #endif
    //srand(time(NULL));
    res.resize(cluster_center * dimension);
    vector<int> cluster_num(cluster_center);
    vector<T> res_offset(cluster_center);
    vector<T> res_tmp(cluster_center * dimension);
    memset(res.data(), 0, cluster_center * dimension * sizeof(T));
    memset(res_tmp.data(), 0, cluster_center * dimension * sizeof(T));
    memset(cluster_num.data(), 0, cluster_center * sizeof(int));
    for (int i = 0; i < cluster_center; ++i){
        int id = rand() % n;
        #ifdef DEBUG_KMEANS
            //std::cout << id << " ";
        #endif
        memcpy(res.data() + i * dimension, data + id * dimension, dimension * sizeof(T));
        res_offset[i] = get_offset<T, dist_type>(res.data() + 1LL * i * dimension, dimension);
    }
    matrix_mul<T>* mm;
    pair<T, idx_t>* dis;
    mm = new rapid_matrix_mul<T>;
    int max_batch = 64;
    mm->set(dimension, 1, max_batch, cluster_center);
    int i, flag;
    for (i = 0; i < 5; ++i){
        #ifdef DEBUG_KMEANS
        T tot_dis = 0;
        std::cout<< "[kmeans] cycle = " << i << std::endl;
        #endif
        memset(cluster_num.data(), 0, cluster_center * sizeof(int));
        memcpy(res_tmp.data(), res.data(), cluster_center * dimension * sizeof(T));
        memset(res.data(), 0, cluster_center * dimension * sizeof(T));
        for (int j = 0; j < n; j += max_batch){
            int size = std::min(max_batch, n - j);
            mm->mul(data + j * dimension, res_tmp.data(), res_offset.data(), size, cluster_center, &dis);
            //std::cout<< "[kmeans] ? " << std::endl;
            for (int k = 0; k < size; ++k){
                cluster_num[dis[k].second]++;
                for (int l = 0; l < dimension; ++l){
                    res[dis[k].second * dimension + l] += data[1LL * (j + k) * dimension + l];
                }
                #ifdef DEBUG_KMEANS
                T v = vec_dis<T, dist_type>(data + (j + k) * dimension, res_tmp.data() + dis[k].second * dimension, dimension);
                tot_dis += v;
                //std::cout << tot_dis << " " << v  << std::endl;
                #endif
            }
        }
        
        flag = 0;
        for (int j = 0; j < cluster_center; ++j){
            if (cluster_num[j] == 0){
                #ifdef DEBUG_KMEANS
                    //std::cout<< " random " << " ";
                #endif
                int id = rand() % n;
                memcpy(res.data() + j * dimension, data + id * dimension, dimension * sizeof(T));
            }
            else{
                for (int k = 0; k < dimension; ++k){
                    res[j * dimension + k] /= cluster_num[j];
                }
            }
        }
        if (dist_type == rsearch::COSINE)
            norm(res.data(), cluster_center, dimension);
        for (int j = 0; j < cluster_center; ++j)
            res_offset[j] = get_offset<T, dist_type>(res.data() + j * dimension, dimension);
        #ifdef DEBUG_KMEANS
        
        std::cout << "tot dis = " << tot_dis << std::endl;
        #endif
        for (int j = 0; j < cluster_center * dimension; ++j) 
        if (res[j] != res_tmp[j]){
            flag = 1;
            break;
        }
        if (flag == 0) break;
    }
    printf("[kmeans]train cycle: %d\n", i);
    return 0;
}
template int k_means<float, COSINE>(const float* data, const int n, const int cluster_center, const int dimension, std::vector<float>& res);
template int k_means<float, EUCLIDEAN>(const float* data, const int n, const int cluster_center, const int dimension, std::vector<float>& res);


}