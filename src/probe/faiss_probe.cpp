#include "probe/faiss_probe.h"
#include "faiss/Index.h"
#include <sys/time.h>
namespace rsearch{
template<typename T>
faiss_probe<T>::faiss_probe(int dimension, int topk, MethodType method_type):probe<T>(){
    this->dimension = dimension;
    this->topk = topk;
    this->method_type = method_type;

}
template faiss_probe<float>::faiss_probe(int, int, MethodType);

template<typename T>
faiss_probe<T>::~faiss_probe(){

}
template faiss_probe<float>::~faiss_probe();

template<typename T>
int faiss_probe<T>::create_gallery(gallery<T> ** ga_ptr){
    faiss_gallery<T> * ga = new faiss_gallery<T>(this->dimension, this->method_type);
    (*ga_ptr) = (gallery<T>*)ga;
    return 0;
}
template int faiss_probe<float>::create_gallery(gallery<float> ** ga_ptr);

template<typename T>
int faiss_probe<T>::query(const T * const x, const int n, gallery<T> * ga, Tout *sims, idx_t *idx){

    faiss_gallery<T>* c_ga = (faiss_gallery<T>*)ga;
    std::vector<faiss::Index::idx_t> labels(n * this->topk);
    c_ga->m_index->search(n, x, this->topk, sims, labels.data());
    for (int i = 0; i < n * this->topk; ++i)
        idx[i] = labels[i];
    return 0;
}
template int faiss_probe<float>::query(const float * const x, const int n, gallery<float> * ga, float *sims, idx_t *idx);

template<typename T>
int faiss_probe<T>::query_with_uids(const T* const x, const int n, gallery<T> * ga, idx_t *uids, const int m, Tout *sims, idx_t *idx){
    return NO_SUPPORT;
}
template int faiss_probe<float>::query_with_uids(const float* const x, const int n, gallery<float> * ga, idx_t *uids, const int m, float *sims, idx_t *idx);

}
