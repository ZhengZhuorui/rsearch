#include "probe/rsearch_probe.h"
namespace rsearch{
template<typename T>
probe<T>* create_probe(const int dimension, const int topk, DistanceType dist_type, MethodType method_type){
    probe<T>* r = NULL;
    switch(method_type){
        case DUMMY:
            if (dist_type == COSINE)
                r = new cpu_base_probe<T, COSINE, base_matrix_mul<T> >(dimension, topk);
            if (dist_type == EUCLIDEAN)
                r = new cpu_base_probe<T, EUCLIDEAN, base_matrix_mul<T> >(dimension, topk);
            break;
        case X86_RAPID:
            if (dist_type == COSINE)
                r = new cpu_base_probe<T, COSINE, rapid_matrix_mul<T> >(dimension, topk);
            if (dist_type == EUCLIDEAN)
                r = new cpu_base_probe<T, EUCLIDEAN, base_matrix_mul<T> >(dimension, topk);
            break;
        case X86_PQIVF:
            if (dist_type == COSINE)
                r = new pqivf_probe<T, COSINE>(dimension, topk);
            if (dist_type == EUCLIDEAN)
                r = new pqivf_probe<T, EUCLIDEAN>(dimension, topk);
            break;
        case X86_RAPID_MULTI_THREAD:
            if (dist_type == COSINE)
                r = new cpu_base_mt_probe<T, COSINE, rapid_matrix_mul<T> >(dimension, topk);
            if (dist_type == EUCLIDEAN)
                r = new cpu_base_mt_probe<T, EUCLIDEAN, rapid_matrix_mul<T> >(dimension, topk);
            break;
        case X86_PQIVF_MULTI_THREAD:
            if (dist_type == COSINE)
                r = new pqivf_mt_probe<T, COSINE>(dimension, topk);
            if (dist_type == EUCLIDEAN)
                r = new pqivf_mt_probe<T, EUCLIDEAN>(dimension, topk);
            break;
        //faiss
        default:
            if (is_same_type<T, float>() == true && dist_type == EUCLIDEAN){
                faiss_probe<float>* rt = new faiss_probe<float>(dimension, topk, method_type);
                r = (probe<T>*)rt;
            }
            else {
                r = NULL;
            }
            break;
    }
    return r;
}
template probe<float>* create_probe(const int dimension, const int topk, DistanceType dist_type, MethodType method_type);
template probe<int8_t>* create_probe(const int dimension, const int topk, DistanceType dist_type, MethodType method_type);

}
