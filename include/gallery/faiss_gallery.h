#pragma once 
#include "gallery/rsearch_gallery.h"
#include "faiss/Index.h"
namespace rsearch {

template<typename T>
class faiss_gallery: public gallery<T> {
public:

    faiss_gallery (int dimension, MethodType method_type);

    virtual ~faiss_gallery() override;

    virtual int init() override ; 

    virtual int add(const T * const x, const int n) override ;

    virtual int add_with_uids(const T * const x, const idx_t * const uid, const int n) override ;

    virtual int change_by_uids(const T * const x, const idx_t * const uid, const int n) override ;

    virtual int remove_by_uids(const idx_t * const uid, const int n) override ;

    virtual int query_by_uids(const idx_t * const uid, int n, T * x) override ;

    virtual int reset() override ;

    virtual int train(const float* const x, int n, int dimension);

    virtual bool have_train(){return this->have_train_;}

    virtual int store_data(std::string file_name);

    virtual int load_data(std::string file_name);


private:

    std::mutex m_mutex;

    //faiss::gpu::StandardGpuResources m_gpu_res;
    int dimension;
    MethodType method_type;

    faiss::Index * m_index;
    faiss::Index * quantizer;

    bool have_train_;
    friend faiss_probe<T>;

};

}