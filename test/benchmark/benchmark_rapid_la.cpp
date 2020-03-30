
#include "matrix/matrix_mul.h"
#include "matrix/rapid_matrix_la.h"
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
int code_len = 32;
int code_per_dimension = 256;
int nIter = 10;
template<typename T>
void get_index(vector<T> idx, int n){
    for (int i = 0; i < n; ++i)
        idx[i] = rand() % code_per_dimension;
}

template<typename T>
void test_perf(){
    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    std::string type_name = rsearch::GetTypeName<T>();
    vector<T> code_book(n * code_len * code_per_dimension);
    vector<int32_t> index(n * code_len);
    get_data(index, code_len, n);
    get_data(code_book, n * code_len * code_per_dimension);
    vector<T> ans(n);
    //rsearch::matrix_la<T>* mm = new rsearch::rapid_matrix_la<T>;
    
    int64_t Bytes = 1LL * n * m;
    //mm->set(dimension, 128, n, m);
    pair<T, int>* res;
    gettimeofday(&time1, &zone);
    for (int i = 0; i < nIter; ++i){
        //mm->mul(a.data(), b.data(), offset.data(), n, m, &res);
        //rsearch::r_dot_prod<T>(a.data(), b.data(), offset.data(), n, m, dimension, res_vec.data(), m);
        rsearch::r_ld_add(code_book.data(), index.data(), ans.data(), n, m, code_len, code_per_dimension * code_len);
    }
    gettimeofday(&time2, &zone);
    float delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
    //float gbytes = Bytes / 1024.0 / 1024.0 / 1024.0 / (delta/nIter) * 1000;
    float gbytes = Bytes / 1000.0 / 1000.0 / 1000.0 / (delta / nIter) * 1000;
    printf("BENCHMARK [%s]: %.4fms, calc : %.2fB\n", type_name.c_str(), delta, gbytes);
    Tout ans_tmp = 0;
    for (int k = 0; k < code_len; ++k)
        ans += code_book[index[k]];
    if (ans_tmp != ans[0]){
        std::cout << "Error! expect: "<< ans_tmp << "result" << ans[0] << std::endl;
    }
    
    //if (res_vec[0] != ans){
    //    std::cout << "Error! expect: "<< ans << "result" << res_vec[0] << std::endl;
    //}
}

int main(){
    
    test_perf<int>();
    test_perf<float>();
    //free(a);
    //free(b);
    //free(offset);
    
}
