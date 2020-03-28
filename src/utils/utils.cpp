#include "utils/utils.h"
namespace rsearch{

void __get_random_data(float* data, int n, int dimension){
    ofstream fout;
    int status;
    char *type_name = abi::__cxa_demangle(typeid(float).name(), NULL,  NULL, &status);
    char fname[200];
    sprintf(fname, "/home/zhengzhuorui/project/data/.rsearch.%s.%d.%d.bin", type_name, dimension, n);
    
    if (file_exist(fname)){
        //printf("[__get_random_data] target 1, file = %s\n", fname);
        ifstream fin(fname, ifstream::binary);
        r_file2bytes<float>(fin, data, n, dimension);
    } else {
        //printf("[__get_random_data] target 2, file = %s\n", fname);
        fout.open(fname, ofstream::binary);
        init_random(data, n, dimension);
        r_bytes2file<float>(fout, data, n, dimension);
    }
}

template<typename T,
        DistanceType dist_type>
void get_random_data(vector<T>& data, int n, int dimension){
    data.resize(n * dimension);
    std::vector<float> data_float;
    data_float.resize(n * dimension);
    __get_random_data(data_float.data(), n, dimension);
    norm(data_float.data(), n, dimension);
    if (is_same_type<T, float>() == true)
        memcpy(data.data(), data_float.data(), 1LL * n * dimension * sizeof(float));
    else{
        float_7bits(data_float.data(), (int8_t*)data.data(), 1LL * n * dimension);
        //for (int i = 0; i < dimension; ++i)
            //std::cout << (int)data[i] << " " << std::endl;
    }
}
template void get_random_data<float, COSINE>(vector<float>& data, int n, int dimension);
template void get_random_data<int8_t, COSINE>(vector<int8_t>& data, int n, int dimension);
template void get_random_data<float, EUCLIDEAN>(vector<float>& data, int n, int dimension);
template void get_random_data<int8_t, EUCLIDEAN>(vector<int8_t>& data, int n, int dimension);


}
