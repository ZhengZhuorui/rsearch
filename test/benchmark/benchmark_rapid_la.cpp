
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
int nIter = 1000;
template<typename T>
void get_index(vector<T> idx, int n, int mo){
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
    vector<int32_t> index(m * code_len);
    
    for (int i = 0; i < m; ++i)
        for (int j = 0, k = 0; j < code_len; ++j, k += code_per_dimension)
            index[i * code_len + j] = k + j % code_per_dimension;
    for (int i = 0; i < n * code_len * code_per_dimension; ++i)
        code_book[i] = i % code_per_dimension;
    vector<T> ans(n * m);
    //rsearch::matrix_la<T>* mm = new rsearch::rapid_matrix_la<T>;
    
    //int64_t Bytes = 1LL * n * m;
    //mm->set(dimension, 128, n, m);
    //pair<T, int>* res;
    gettimeofday(&time1, &zone);
    std::cout << "target 1" << std::endl;
    for (int i = 0; i < nIter; ++i){
        //mm->mul(a.data(), b.data(), offset.data(), n, m, &res);
        //rsearch::r_dot_prod<T>(a.data(), b.data(), offset.data(), n, m, dimension, res_vec.data(), m);
        rsearch::r_ld_add(code_book.data(), index.data(), ans.data(), n, m, code_len, code_per_dimension * code_len, m);
    }
    gettimeofday(&time2, &zone);
    std::cout << "target 2 " << std::endl;
    float delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
    //float gbytes = Bytes / 1024.0 / 1024.0 / 1024.0 / (delta/nIter) * 1000;
    float QPS = 1.0 * n * m / (delta / nIter);
    printf("BENCHMARK [%s]: %.4fms, calc : %.2fB\n", type_name.c_str(), delta, QPS);
    T ans_tmp = 0;
    
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < m; ++j){
            ans_tmp = 0;
            for (int k = 0; k < code_len; ++k)
                ans_tmp += code_book[i * code_per_dimension * code_len + index[j * code_len + k] - k * code_per_dimension];
            //std::cout << ans[0] << " " << ans[i * m + j] << std::endl;
            if (ans_tmp != ans[i * m + j]){
                std::cout << "Error! " << i << " " << j << " expect: "<< ans_tmp << ", result" << ans[i * m + j] << std::endl;
                return; 
            }
        }
    }
    /*for (int k = 0; k < code_len; ++k)
        ans_tmp += code_book[index[k] - k * code_per_dimension];
    std::cout << ans_tmp << " " << ans[0] << std::endl;*/
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
