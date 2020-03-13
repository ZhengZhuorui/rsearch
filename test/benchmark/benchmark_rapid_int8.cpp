
#include "matrix/matrix_mul.h"
#include "matrix/rapid_matrix_mul.h"
#include "rsearch_type.h"
#include <sys/sysinfo.h>
#include <sys/time.h>
using namespace std;

int n = 32;
int m = 8192;
int dimension = 512;
int nIter = 100;
template<typename T>
void get_data(T* a, int n, int d){
    memset(a, 0, n * d * sizeof(T));
    for (int i = 0; i < n; ++i)
        a[i * d] = i % 126 - 63;
}

int main(){
    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    int8_t *a = (int8_t*)malloc(n * dimension);
    int8_t *b = (int8_t*)malloc(m * dimension);
    int *offset = (int*)malloc(m);
    memset(offset, 0, m * sizeof(int));
    get_data(a, n, dimension);
    get_data(b, m, dimension);
    rsearch::matrix_mul<int8_t>* mm = new rsearch::rapid_matrix_mul<int8_t>;
    mm->set(dimension, 128, n, m);
    long OPS = long(n) * long(m) * long(dimension) * 2;
    long BBytes = long(m) * long(dimension) * sizeof(int8_t); 
    long CBytes = long(m) * long(n) * sizeof(int8_t);
    long OBytes = long(m) * sizeof(int8_t);
    long Bytes = BBytes + CBytes + OBytes;
    pair<int, int>* res;
    gettimeofday(&time1, &zone);
    for (int i = 0; i < nIter; ++i)
    mm->mul(a, b, offset, n, m, &res);
    gettimeofday(&time2, &zone);
    float delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
    float gbytes = Bytes / 1024.0 / 1024.0 / 1024.0 / (delta/nIter) * 1000;
    printf("BENCHMARK: %.4fms, Bandwidth: %.2f GB", delta, Bytes);
    
    free(a);
    free(b);
    free(offset);
    
}