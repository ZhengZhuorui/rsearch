#include "gallery/cpu_base_gallery.h"
namespace rsearch{
template<typename T,
        DistanceType dist_type>
cpu_base_gallery<T, dist_type>::cpu_base_gallery(int dimension) : gallery<T, dist_type>(){
    this->dimension = dimension;
    this->num = 0;
    this->max_id = 0;
}
template cpu_base_gallery<int8_t, COSINE>::cpu_base_gallery(int);
template cpu_base_gallery<float, COSINE>::cpu_base_gallery(int);
template cpu_base_gallery<int8_t, EUCLIDEAN>::cpu_base_gallery(int);
template cpu_base_gallery<float, EUCLIDEAN>::cpu_base_gallery(int);

template<typename T,
        DistanceType dist_type>
cpu_base_gallery<T, dist_type>::~cpu_base_gallery(){
    this->data.clear();
    this->index.clear();
    this->ids.clear();
    this->offset.clear();
}
template cpu_base_gallery<int8_t, COSINE>::~cpu_base_gallery();
template cpu_base_gallery<float, COSINE>::~cpu_base_gallery();
template cpu_base_gallery<int8_t, EUCLIDEAN>::~cpu_base_gallery();
template cpu_base_gallery<float, EUCLIDEAN>::~cpu_base_gallery();

template<typename T,
        DistanceType dist_type>
int cpu_base_gallery<T, dist_type>::init(){
    this->data.clear();
    this->index.clear();
    this->ids.clear();
    this->offset.clear();
    this->num = 0;
    return 0;
}
template int cpu_base_gallery<int8_t, COSINE>::init();
template int cpu_base_gallery<float, COSINE>::init();
template int cpu_base_gallery<int8_t, EUCLIDEAN>::init();
template int cpu_base_gallery<float, EUCLIDEAN>::init();

template<typename T,
        DistanceType dist_type>
int cpu_base_gallery<T, dist_type>::add(const T* const x, const int n){
    this->mtx.lock();
    this->data.reserve((this->num + n) * this->dimension);
    memcpy(this->data.data() + 1LL * this->num * this->dimension, x, 1LL * sizeof(T) * n * this->dimension);
    this->offset.reserve((this->num+n));
    for (int i = 0; i < n; ++i){
        this->index[this->max_id] = this->num + i;
        this->ids.push_back(this->max_id);
        this->max_id++;
    }
    for (int i = 0 ; i < n ; ++i){
        this->offset[this->num + i] = get_offset<T, dist_type>(this->data.data() + 1LL * (this->num + i) * this->dimension,  this->dimension);
    }    
    this->num += n;
    this->mtx.unlock();
    return 0;
}
template int cpu_base_gallery<int8_t, COSINE>::add(const int8_t* const x, const int n);
template int cpu_base_gallery<float, COSINE>::add(const float* const x, const int n);
template int cpu_base_gallery<int8_t, EUCLIDEAN>::add(const int8_t* const x, const int n);
template int cpu_base_gallery<float, EUCLIDEAN>::add(const float* const x, const int n);

template<typename T,
        DistanceType dist_type>
int cpu_base_gallery<T, dist_type>::add_with_uids(const T* const x, const idx_t * const uids, const int n){
    this->mtx.lock();
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) != this->index.end()){
            return INDEX_EXISTS;
        }
    }
    for (int i = 0; i < n; ++i){
        this->ids.push_back(uids[i]);
        this->index[uids[i]] = num;
        this->max_id = std::max(this->max_id, uids[i] + 1);
    }
    
    this->data.reserve((this->num + n) * this->dimension);
    memcpy(this->data.data() + 1LL * this->num * this->dimension, x, 1LL * sizeof(T) * n * this->dimension);
    this->offset.reserve(this->num + n);
    for (int i = 0 ; i < n ; ++i){
        this->offset[this->num + i] = get_offset<T, dist_type>(this->data.data() + 1LL * (this->num + i) * this->dimension, this->dimension);
    }  
    this->num += n;
    this->mtx.unlock();
    return 0;
}
template int cpu_base_gallery<int8_t, COSINE>::add_with_uids(const int8_t* const x, const idx_t * const uids, const int n);
template int cpu_base_gallery<float, COSINE>::add_with_uids(const float* const x, const idx_t * const uids, const int n);
template int cpu_base_gallery<int8_t, EUCLIDEAN>::add_with_uids(const int8_t* const x, const idx_t * const uids, const int n);
template int cpu_base_gallery<float, EUCLIDEAN>::add_with_uids(const float* const x, const idx_t * const uids, const int n);

template <typename T,
        DistanceType dist_type>
int cpu_base_gallery<T, dist_type>::change_by_uids(const T* const x, const idx_t * const uids, const int n){
    this->mtx.lock();
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) == this->index.end())
            return INDEX_NO_FIND;
    }
    for (int i = 0; i < n; ++i){
        memcpy(this->data.data() + 1LL * this->index[uids[i]] * this->dimension, x + 1LL * i * dimension, sizeof(T) * this->dimension);
        this->offset[this->index[uids[i]]] = get_offset<T, dist_type>(this->data.data() + 1LL * this->index[uids[i]] * this->dimension, this->dimension);
    }
    this->mtx.unlock();
    return 0;
}
template int cpu_base_gallery<int8_t, COSINE>::change_by_uids(const int8_t* const x, const idx_t * const uids, const int n);
template int cpu_base_gallery<float, COSINE>::change_by_uids(const float* const x, const idx_t * const uids, const int n);
template int cpu_base_gallery<int8_t, EUCLIDEAN>::change_by_uids(const int8_t* const x, const idx_t * const uids, const int n);
template int cpu_base_gallery<float, EUCLIDEAN>::change_by_uids(const float* const x, const idx_t * const uids, const int n);

template<typename T,
        DistanceType dist_type>
int cpu_base_gallery<T, dist_type>::remove_by_uids(const idx_t* const uids, const int n){
    this->mtx.lock();
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) == this->index.end())
            return INDEX_NO_FIND;
    }
    for (int i = 0; i < n ; ++i){
        int p = this->index[uids[i]];
        memcpy(this->data.data() + 1LL * p * this->dimension, this->data.data() + 1LL * (this->num - 1) * this->dimension, sizeof(T) * this->dimension);
        this->offset[p] = this->offset[num - 1];
        this->index[this->ids[this->num - 1]] = p;
        this->ids[p] = this->ids[this->num - 1];
        this->index.erase(uids[i]);
        this->ids.pop_back();
        this->num--;
    }
    this->data.reserve(this->num * this->dimension);
    this->offset.reserve(this->num);
    this->mtx.unlock();
    return 0;   
}
template int cpu_base_gallery<int8_t, COSINE>::remove_by_uids(const idx_t * const uids, const int n);
template int cpu_base_gallery<float, COSINE>::remove_by_uids(const idx_t * const uids, const int n);
template int cpu_base_gallery<int8_t, EUCLIDEAN>::remove_by_uids(const idx_t * const uids, const int n);
template int cpu_base_gallery<float, EUCLIDEAN>::remove_by_uids(const idx_t * const uids, const int n);

template<typename T,
        DistanceType dist_type>
int cpu_base_gallery<T, dist_type>::query_by_uids(const idx_t* const uids, const int n, T * x){
    for (int i = 0; i < n; ++i){
        if (this->index.find(uids[i]) == this->index.end())
            return INDEX_NO_FIND;
    }
    for (int i = 0; i < n ; ++i){
        memcpy(x, &this->data[this->index[uids[i]]], sizeof(T) * this->dimension);
    }
    return 0;
}
template int cpu_base_gallery<int8_t, COSINE>::query_by_uids(const idx_t * const uids, const int n, int8_t* x);
template int cpu_base_gallery<float, COSINE>::query_by_uids(const idx_t * const uids, const int n, float* x);
template int cpu_base_gallery<int8_t, EUCLIDEAN>::query_by_uids(const idx_t * const uids, const int n, int8_t* x);
template int cpu_base_gallery<float, EUCLIDEAN>::query_by_uids(const idx_t * const uids, const int n, float* x);

template<typename T,
        DistanceType dist_type>
int cpu_base_gallery<T, dist_type>::reset(){
    this->data.clear();
    this->index.clear();
    this->ids.clear();
    this->offset.clear();
    this->num = 0;
    return 0;
}
template int cpu_base_gallery<int8_t, COSINE>::reset();
template int cpu_base_gallery<float, COSINE>::reset();
template int cpu_base_gallery<int8_t, EUCLIDEAN>::reset();
template int cpu_base_gallery<float, EUCLIDEAN>::reset();

}