#include "rsearch.h"

namespace rsearch{
rsearch::rsearch(){
    this->probe_float = NULL;
    this->probe_int = NULL;
}
rsearch::rsearch(const int dimension, const int topk, DistanceType dist_type, MethodType method_type, VarType var_type){
    this->var_type = var_type;
    this->probe_float = NULL;
    this->probe_int = NULL;
    switch (var_type)
    {
    case FLOAT32:
        this->probe_float = create_probe<float>(dimension, topk, dist_type, method_type);
        this->probe_float->create_gallery(&this->ga_float);
        break;
    case INT8:
        this->probe_int = create_probe<int8_t>(dimension, topk, dist_type, method_type);
        this->probe_int->create_gallery(&this->ga_int);
        break;
    default:
        break;
    }
    this->dimension = dimension;
    this->topk = topk;
}
rsearch::~rsearch(){
    if (this->probe_float!=NULL){
        delete this->probe_float;
        delete this->ga_float;
    }
    if (this->probe_int!=NULL){
        delete this->probe_int;
        delete this->ga_int;
    }
}

int rsearch::query(const float* x, const int n, float *sims, idx_t *idx){
    return probe_float->query(x, n, ga_float, sims, idx);
}
int rsearch::query(const int8_t* x, const int n,  int *sims, idx_t *idx){
    return probe_int->query(x, n, ga_int, sims, idx);
}
int rsearch::add(const float* x, const int n){
    return ga_float->add(x, n);
}
int rsearch::add(const int8_t* x, const int n){
    return ga_int->add(x, n);
}

int rsearch::add_with_uids(const float* x, const idx_t * uids, const int n){
    return ga_float->add_with_uids(x, uids, n);
}
int rsearch::add_with_uids(const int8_t* x, const idx_t * uids, const int n){
    return ga_int->add_with_uids(x, uids, n);
}

int rsearch::change_by_uids(const float* x, const idx_t * uids, const int n){
    return ga_float->change_by_uids(x, uids, n);
}
int rsearch::change_by_uids(const int8_t* x, const idx_t * uids, const int n){
    return ga_int->change_by_uids(x, uids, n);
}

int rsearch::remove_by_uids(const idx_t * uids, const int n){
    switch (var_type)
    {
    case FLOAT32:
        return this->ga_float->remove_by_uids(uids, n);
        break;
    case INT8:
        return this->ga_int->remove_by_uids(uids, n);
        break;
    default:
        break;
    }
    return 0;
}

int rsearch::query_by_uids(const idx_t* uid, const int n, float * x){
    return this->ga_float->query_by_uids(uid, n, x);
}
int rsearch::query_by_uids(const idx_t* uid, const int n, int8_t * x){
    return this->ga_int->query_by_uids(uid, n, x);
}

int rsearch::reset(){
    switch (var_type)
    {
    case FLOAT32:
        this->ga_float->reset();
        break;
    case INT8:
        this->ga_int->reset();
        break;
    default:
        break;
    }
    return 0;
}

int rsearch::train(const float* x, int n){
    switch (var_type)
    {
    case FLOAT32:
        return this->ga_float->train(x, n);
        break;
    case INT8:
        return this->ga_int->train(x, n);
        break;
    default:
        break;
    }
    return 0;
}

int rsearch::store_data(const char* file_name){
    switch (var_type)
    {
    case FLOAT32:
        return this->ga_float->store_data(file_name);
        break;
    case INT8:
        return this->ga_int->store_data(file_name);
        break;
    default:
        break;
    }
    return 0;
}

int rsearch::load_data(const char* file_name){
    switch (var_type)
    {
        case FLOAT32:
        return this->ga_float->load_data(file_name);
        break;
    case INT8:
        return this->ga_int->load_data(file_name);
        break;
    default:
        break;
    }
    return 0;
}

}
