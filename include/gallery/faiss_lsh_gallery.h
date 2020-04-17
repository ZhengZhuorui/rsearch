#pragma once 
#include "gallery/rsearch_gallery.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>

namespace rsearch {

template<typename T,
        DistanceType dist_type>
class faiss_lsh_gallery: public base_gallery<T> {
public:

    faiss_lsh_gallery (int dimension);

    virtual ~faiss_lsh_gallery() override;


    virtual int init() override ; 

    virtual int add(const T * const x, const int n) override ;

    virtual int add_with_uids(const T * const x, const idx_t * const uid, const int n) override ;

    virtual int change_by_uids(const T * const x, const idx_t * const uid, const int n) override ;

    virtual int remove_by_uids(const idx_t * const uid, const int n) override ;

    virtual int query_by_uids(const idx_t * const uid, int n, T * x) override ;

    virtual int reset() override ;

    virtual int store_data(std::string file_name){
        return NO_SUPPORT;
    }

    virtual int load_data(std::string file_name){
        return NO_SUPPORT;
    }


private:

    std::mutex m_mutex;

    //faiss::gpu::StandardGpuResources m_gpu_res;
    
    faiss::IndexFlat * m_flat_index;

    friend flat_probe<T, DistanceType dist_type>;

};

}