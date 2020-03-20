#include "probe/rsearch_probe.h"
namespace rsearch{
template<typename T, 
        DistanceType dist_type>
probe<T, dist_type>* create_probe(int dimension, int topk, MethodType method_type){
    probe<T, dist_type> * r;
    switch(method_type){
        case DUMMY:
            r = new cpu_base_probe<T, dist_type, base_matrix_mul<T> >(dimension, topk);
            break;
        case X86_RAPID:
            r = new cpu_base_probe<T, dist_type, rapid_matrix_mul<T> >(dimension, topk);
            break;
        case X86_PQIVF:
            r = new pqivf_probe<T, dist_type>(dimension, topk);
            break;
        default:
            r = NULL;
            break;
    }
    return r;
}
template probe<int8_t, COSINE>* create_probe(int dimension, int topk, MethodType method_type);
template probe<float, COSINE>* create_probe(int dimension, int topk, MethodType method_type);
template probe<int8_t, EUCLIDEAN>* create_probe(int dimension, int topk, MethodType method_type);
template probe<float, EUCLIDEAN>* create_probe(int dimension, int topk, MethodType method_type);

}