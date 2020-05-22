#include "utils/utils.h"
namespace rsearch{

void __get_random_data(float* data, int n, int dimension){
    ofstream fout;
    int status;
    char *type_name = abi::__cxa_demangle(typeid(float).name(), NULL,  NULL, &status);
    char fname[200];
    std::string file_name = std::string(DATA_DIR) + "/.rsearch.%s.%d.%d.bin";
    sprintf(fname, file_name.c_str(), type_name, dimension, n);
    if (file_exist(fname)){
        ifstream fin(fname, ifstream::binary);
        r_file2bytes<float>(fin, data, n, dimension);
    } else {
        fout.open(fname, ofstream::binary);
        init_random(data, 1LL * n * dimension);
        r_bytes2file<float>(fout, data, n, dimension);
    }
}

template<typename T,
        DistanceType dist_type>
void get_random_data(vector<T>& data, int n, int dimension){
    std::vector<float> data_float;
    data_float.resize(1LL * n * dimension);
    data.resize(1LL * n * dimension);
    __get_random_data(data_float.data(), n, dimension);
    
    if (dist_type == COSINE)
        norm(data_float.data(), n, dimension);
    if (is_same_type<T, float>() == true)
        memcpy(data.data(), data_float.data(), 1LL * n * dimension * sizeof(float));
    else{
        //std::cout << data_float[0] << std::endl;
        pair<float, float> arg = __float_7bits(data_float.data(), 1LL * n * dimension);
        std::cout << data.size() << std::endl;
        float_7bits(data_float.data(), (int8_t*)data.data(), 1LL * n * dimension, arg.first, arg.second);
        //std::cout << "get random data: " << (int)data[0] << " " << (int)data[2200000001LL] << std::endl;
    }
}
template void get_random_data<float, COSINE>(vector<float>& data, int n, int dimension);
template void get_random_data<int8_t, COSINE>(vector<int8_t>& data, int n, int dimension);
template void get_random_data<float, EUCLIDEAN>(vector<float>& data, int n, int dimension);
template void get_random_data<int8_t, EUCLIDEAN>(vector<int8_t>& data, int n, int dimension);


}
