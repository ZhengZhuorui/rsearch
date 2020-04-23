#pragma once
#include "rsearch_def.h"
#include "probe/rsearch_probe.h"
#include "gallery/faiss_gallery.h"
namespace rsearch{
using std::pair;
using std::vector;
using std::make_pair;
template<typename T>
class faiss_probe : public probe<T>{
public:
    faiss_probe(int dimension, int topk, MethodType method_type);
    ~faiss_probe();
    using Tout = typemap_t<T>;
    virtual int create_gallery(gallery<T> ** ga_ptr) override;
    virtual int query(const T * const x, const int n, gallery<T> * ga, Tout *sims, idx_t *idx) override;
    virtual int query_with_uids(const T* const x, const int n, gallery<T> * ga, idx_t *uids, const int m, Tout *sims, idx_t *idx) override;
private:
    int dimension, topk;
    MethodType method_type;
};

}