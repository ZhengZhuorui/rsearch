
#include "matrix/matrix_mul.h"
#include "matrix/rapid_matrix_mul.h"
#include "matrix/base_matrix_mul.h"
#include <utils/avx2_asm.h>
#include "rsearch_type.h"
#include <utils/utils.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <utils/helpers.h>
using std::pair;
using std::vector;
int n = 32;
int m = 8192;
int dimension = 512;
int nIter = 100;
void get_data(vector<int8_t>&data, int n, int dimension){
    data.resize(n * dimension);
    memset(data.data(), 0, n * dimension * sizeof(int8_t));
    for (int i = 0; i < n; ++i) data[i * dimension] = i % 63;
}
void get_data(vector<float>&data, int n, int dimension){
    data.resize(n * dimension);
    memset(data.data(), 0, n * dimension * sizeof(float));
    for (int i = 0; i < n; ++i) data[i * dimension] = (i % 126) * 1.0 / 126;
}
template<typename T>
void test_perf(){
    using Tout = rsearch::typemap_t<T>;
    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    std::string type_name = rsearch::GetTypeName<T>();
    vector<T> a, b;
    vector<Tout> offset(m);
    vector<Tout> res_vec(n*m);
    memset(res_vec.data(), 0, sizeof(Tout) * n * m);
    memset(offset.data(), 0, m * sizeof(Tout));
    rsearch::get_random_data<T, rsearch::COSINE>(a, n, dimension);
    rsearch::get_random_data<T, rsearch::COSINE>(b, m, dimension);
    /*get_data(a, n, dimension);

    for (int i = 0; i < n; ++i)
        std::cout << (Tout)a[i * dimension] << " ";
    std::cout << std::endl;
    get_data(b, m, dimension);*/
    rsearch::matrix_mul<T>* mm = new rsearch::rapid_matrix_mul<T>;
    int64_t Bytes = 1LL * n * m;
    mm->set(dimension, 128, n, m);
    pair<Tout, int>* res;
    gettimeofday(&time1, &zone);
    for (int i = 0; i < nIter; ++i)
        //mm->mul(a.data(), b.data(), offset.data(), n, m, &res);
        rsearch::r_dot_prod<T>(a.data(), b.data(), offset.data(), n, m, dimension, res_vec.data(), m);
        
    gettimeofday(&time2, &zone);
    float delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
    //float gbytes = Bytes / 1024.0 / 1024.0 / 1024.0 / (delta/nIter) * 1000;
    float gbytes = Bytes / 1000.0 / 1000.0 / 1000.0 / (delta / nIter) * 1000;
    printf("BENCHMARK [%s]: %.4fms, calc : %.2fB\n", type_name.c_str(), delta, gbytes);
    Tout ans = 0;
    for (int k = 0; k < dimension; ++k)
        ans += a[k] * b[k];
    
    //if (res_vec[0] != ans){
    //    std::cout << "Error! expect: "<< ans << "result" << res_vec[0] << std::endl;
    //}
    delete mm;
}

int main(){
    
    test_perf<int8_t>();
    test_perf<float>();
    //free(a);
    //free(b);
    //free(offset);
    
}
