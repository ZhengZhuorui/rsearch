#pragma once
#include "rsearch_type.h"
#include "probe/rsearch_probe.h"
#include "gallery/rsearch_gallery.h"
#include "rsearch_def.h"

namespace rsearch{

class rsearch{
public:
    rsearch();
    rsearch(const int dimension, const int topk, DistanceType dist_type, MethodType method_type, VarType var_type);
    ~rsearch();
    
    int query(const float* x, const int n, float *sims, idx_t *idx);
    int query(const int8_t* x, const int n, int *sims, idx_t *idx);
    //int query(const query_form* x, const int n, std::vector<idx_t> &idx);
    
    int query_with_uids(const float* const x, const int n, idx_t *uids, const int m, float *sims, idx_t *idx){
        return NO_SUPPORT;
    }
    int query_with_uids(const int8_t* const x, const int n, idx_t *uids, const int m, int *sims, idx_t *idx){
        return NO_SUPPORT;
    }

    int add(const float* x, const int n);
    int add(const int8_t* x, const int n);

    int add_with_uids(const float* x, const idx_t * uids, const int n);
    int add_with_uids(const int8_t* x, const idx_t * uids, const int n);

    int change_by_uids(const float* x, const idx_t * uids, const int n);
    int change_by_uids(const int8_t* x, const idx_t * uids, const int n);

    int remove_by_uids(const idx_t * uids, const int n);

    int query_by_uids(const idx_t* uid, const int n, float * x);
    int query_by_uids(const idx_t* uid, const int n, int8_t * x);

    int reset();

    int train(const float* x, int n);

    int store_data(std::string file_name);

    int load_data(std::string file_name);

    int dimension, topk;

    VarType var_type;
    MethodType method_type;

private:
    probe<float>* probe_float;
    probe<int8_t>* probe_int;

    gallery<float>* ga_float;
    gallery<int8_t>* ga_int;
    
};

}