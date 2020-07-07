#include <utils/ThreadPool.h>
#include "unit_test.h"
static void add(int *a, int *b, int *c, int n,int K){
    static int m = 0;
    //std::cout << "[add]" << m++ << " " << std::endl;
    for (int i = 0; i < n; ++i)
    for (int j = 0; j < K; ++j)
        c[i * K + j] = a[i * K + j] + b[i * K + j];
}

int test_thread_pool(int n, int batch, int K){
    std::vector<int> A, B, C, C_tmp;
    A.resize(n * K);
    B.resize(n * K);
    C.resize(n * K);
    C_tmp.resize(n * K);
    for (int i = 0; i < n * K; ++i){
        A[i] = rand() % 10000;
        B[i] = rand() % 10000;
    } 
    for (int i = 0; i < n; ++i)
    for (int j = 0; j < K; ++j){
        C_tmp[i * K + j] = A[i * K + j] + B[i * K + j];
    }
    rsearch::ThreadPool thread_pool;
    std::cout << "[multi_thread] start 0." << std::endl;
    thread_pool.start();
    std::cout << "[multi_thread] start 1." << std::endl;
    int i;
    for (i = 0; i < n / 2; i += batch){
        std::function<void()> f = std::bind(add, A.data() + i * K, B.data() + i * K, C.data() + i * K, std::min(batch, n - i), K);
        thread_pool.add_task(f);
    }

    std::cout << "[multi_thread] running." << std::endl;
    thread_pool.synchronize();
    std::cout << "[multi_thread] running 1." << std::endl;
    for (; i < n; i += batch){
        std::function<void()> f = std::bind(add, A.data() + i * K, B.data() + i * K, C.data() + i * K, std::min(batch, n - i), K);
        thread_pool.add_task(f);
    }
    thread_pool.synchronize();
    std::cout << "[multi_thread] running 2." << std::endl;
    thread_pool.stop();
    std::cout << "[multi_thread] end." << std::endl;
    int flag = 0;
    for (int i = 0; i < n * K; ++i)
    if (C[i] != C_tmp[i]){
        std::cout << "Expect " << i << " : " << C_tmp[i] << ", result: " << C[i] << std::endl;
        flag = 1;
    }
    return flag;

}

TEST_F(UnitTest, ThreadPoolTest) {
    //EXPECT_EQ(0, test_thread_pool(10000, 100, 100) );
    
}