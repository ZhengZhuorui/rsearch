#include <rsearch_type.h>
#include <gallery/rsearch_gallery.h>
#include <probe/rsearch_probe.h>
#include <utils/utils.h>
#include <utils/helpers.h>
#include <sys/time.h>
#include "unit_test.h"

template<typename T,
        rsearch::DistanceType dist_type>
int test_query(const int N, const int K, const int Dimension, const rsearch::MethodType mt){
    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    std::string target_name;
    std::string type_name;
    std::string file_name;
    target_name = rsearch::GetMethodName(mt);
    type_name = rsearch::GetTypeName<T>();

    file_name = "rsearch_gallery." + rsearch::GetGalleryName(mt) + "." + std::to_string(N) + "." + std::to_string(Dimension) + "." + \
                type_name + ".bin";
    std::cout << file_name << std::endl;
    rsearch::probe<T, dist_type>* probe = rsearch::create_probe<T, dist_type>(Dimension, K, mt);
    std::cout << "[test_query] target 1\n" << std::endl; 
    rsearch::gallery<T, dist_type>* ga;
    int ret = probe->create_gallery(&ga);
    std::cout << "[test_query] target 2\n" << std::endl; 
    if (ret != 0){
        printf("Create probe failed.\n");
        return -1;
    }
    std::vector<T> x;
    if (ga->init() != 0){
        printf("Gallery init failed.\n");
        return -1;
    }
    std::cout << "[test_query] target 3\n" << std::endl; 
    if (rsearch::file_exist(file_name.c_str()) == false){
        std::cout << file_name << std::endl;
        rsearch::get_random_data<T, dist_type>(x, N, Dimension);
        ret = ga->add(x.data(), N);
        ga->store_data(file_name);
    }
    else{
        ga->load_data(file_name);
    }
    std::cout << "[test_query] target 4\n" << std::endl; 
    const int batch = 128;
    using Tout = rsearch::typemap_t<T>;
    Tout sims[K * batch];
    uint32_t top_uids[K * batch];//, real_uids[K * batch];
    int target = 1123;
    //for (int i = 0; i < batch; ++i)
    //    real_uids[i] = target + i;
    const int nIter = 1;
    gettimeofday(&time1, &zone);
    for (int i = 0; i < nIter; ++i)
        ret = probe->query(&x[target * Dimension], batch, ga, &sims[0], &top_uids[0]);
    gettimeofday(&time2, &zone);
    std::cout << "[test_query] target 5\n" << std::endl; 
    if (ret != 0){
        printf("Query Faieled\n");
        return -1;
    }
    float delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;

    float QPS = 1000 / (delta / nIter) * batch;

    printf("[%s] <%s> Batch [%3d] Query On %8d Gallery, cost:%4fms, QPS: %.2f\n", target_name.c_str(), type_name.c_str(), batch, N, delta/nIter, QPS);
    int flag = 0;
    for (int i = 0; i < batch; ++i){
        if (top_uids[i * K] != target + i){
            std::cout << "Expect uid: " << target + i << ", real uid: "  << top_uids[i * K] << ", real sims:" << sims[i * K]<< std::endl;
            flag = -1;
        }
    }
    delete probe;
    delete ga;
    return flag;
}

TEST_F(UnitTest, QueryPerfTest) {
    
    EXPECT_EQ(0, (test_query<float, rsearch::COSINE>(30000, 128, 512, rsearch::DUMMY)) );
    EXPECT_EQ(0, (test_query<int8_t, rsearch::COSINE>(30000, 128, 512, rsearch::DUMMY)) );
    //EXPECT_EQ(0, (test_query<float, rsearch::COSINE>(30000, 128, 512, rsearch::X86_RAPID)) );
    //EXPECT_EQ(0, (test_query<int8_t, rsearch::COSINE>(30000, 128, 512, rsearch::X86_RAPID)) );
    //EXPECT_EQ(0, (test_query<float, rsearch::EUCLIDEAN>(30000, 128, 512, rsearch::X86_RAPID)) );
    //EXPECT_EQ(0, (test_query<int8_t, rsearch::EUCLIDEAN>(30000, 128, 512, rsearch::X86_RAPID)) );
    EXPECT_EQ(0, (test_query<float, rsearch::COSINE>(30000, 128, 512, rsearch::X86_PQIVF)) );
    EXPECT_EQ(0, (test_query<int8_t, rsearch::COSINE>(30000, 128, 512, rsearch::X86_PQIVF)) );
}
