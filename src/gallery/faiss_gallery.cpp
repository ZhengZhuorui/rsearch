#include "gallery/faiss_gallery.h"
#include "utils/helpers.h"
#include "utils/utils.h"
#include "faiss/Index.h"
#include "faiss/IndexLSH.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexHNSW.h"


namespace rsearch{
using std::ofstream;
using std::ifstream;
using std::unordered_map;
using std::vector;
using std::pair;
template<typename T>
faiss_gallery<T>::faiss_gallery(int dimension, MethodType method_type) : gallery<T>(){
    this->method_type = method_type;
    this->dimension = dimension;
    switch(method_type){
        case FAISS_LSH:
            this->m_index = new faiss::IndexLSH(dimension, 50);
            break;
        case FAISS_FLAT:
            this->m_index = new faiss::IndexFlat(dimension);
            break;
        case FAISS_HNSW:
            this->m_index = new faiss::IndexHNSW(dimension, 5);
            break;
        case FAISS_IVFPQ:
            this->quantizer = new faiss::IndexFlatL2(dimension);
            this->m_index = new faiss::IndexIVFPQ(this->quantizer, dimension, 4096, dimension / 32, 8);
            ((faiss::IndexIVFPQ * )(this->m_index))->nprobe = 1024;
            break;
        default:
            this->m_index = NULL;
            break;
    }

}
template faiss_gallery<float>::faiss_gallery(int, MethodType method_type);

template<typename T>
faiss_gallery<T>::~faiss_gallery(){
    if (this->m_index != NULL)
        delete this->m_index;
}
template faiss_gallery<float>::~faiss_gallery();

template<typename T>
int faiss_gallery<T>::init(){
    if (this->method_type == FAISS_IVFPQ || this->method_type == FAISS_LSH || this->method_type == FAISS_HNSW){
        if (this->have_train_ == false){  
            vector<float> data;
            get_random_data<float, EUCLIDEAN>(data, 200000, this->dimension);
            this->train(data.data(), 200000, this->dimension);
        }
    }
    return 0;
}
template int faiss_gallery<float>::init();

template<typename T>
int faiss_gallery<T>::reset(){
    this->m_index->reset();
    return 0;
}
template int faiss_gallery<float>::reset();

template<typename T>
int faiss_gallery<T>::add(const T* const x, const int n){
    this->m_index->add(n, x);
    return 0;
}
template int faiss_gallery<float>::add(const float* const x, const int n);

template<typename T>
int faiss_gallery<T>::add_with_uids(const T* const x, const idx_t * const uids, const int n){
    vector<faiss::Index::idx_t> xids(n);
    for (int i = 0; i < n; ++i) xids[i] = uids[i];
    this->m_index->add_with_ids(n, x, xids.data());
    return 0;
}
template int faiss_gallery<float>::add_with_uids(const float* const x, const idx_t * const uids, const int n);

template <typename T>
int faiss_gallery<T>::change_by_uids(const T* const x, const idx_t * const uids, const int n){
    return NO_SUPPORT;
}
template int faiss_gallery<float>::change_by_uids(const float* const x, const idx_t * const uids, const int n);

template<typename T>
int faiss_gallery<T>::remove_by_uids(const idx_t* const uids, const int n){
    return NO_SUPPORT;
}
template int faiss_gallery<float>::remove_by_uids(const idx_t * const uids, const int n);

template<typename T>
int faiss_gallery<T>::query_by_uids(const idx_t* const uids, const int n, T * x){
    return NO_SUPPORT;
}
template int faiss_gallery<float>::query_by_uids(const idx_t * const uids, const int n, float* x);

template<typename T>
int faiss_gallery<T>::train(const float* data, const int n, const int dimension){
    this->m_index->train(n, data);
    return 0;
}
template int faiss_gallery<float>::train(const float* data, const int n, const int dimension);

template<typename T>
int faiss_gallery<T>::load_data(std::string file_name){
    return NO_SUPPORT;
}
template int faiss_gallery<float>::load_data(std::string file_name);

template<typename T>
int faiss_gallery<T>::store_data(std::string file_name){
    return NO_SUPPORT;
}
template int faiss_gallery<float>::store_data(std::string file_name);

}
