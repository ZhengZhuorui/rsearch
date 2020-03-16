#include <rsearch_type.h>
#include <gallery/rsearch_gallery.h>
#include <probe/rsearch_probe.h>
#include <utils/utils.h>
#include "unit_test.h"

template<typename T,
        rsearch::DistanceType dist_type>
int test_query(const int N, const int K, const int Dimension, const rsearch::MethodType mt){
    struct timezone zone;
    struct timeval time1;
    struct timeval time2;

    rsearch::probe<T, dist_type>* probe = rsearch::create_probe<T, dist_type>(Dimension, K, mt);
    rsearch::gallery<T, dist_type>* ga;
    int ret = probe->create_gallery(ga);
    if (ret != 0){
        printf("Create probe failed.\n");
        return -1;
    }
    std::vector<T> x;
    x.resize(N * Dimension);
    get_random_data<T>(x.data(), N, Dimension);
    ret = ga->add(x, N);
    const int batch = 128;
    using Tout = rsearch::typemap_t<T>;
    Tout sims[K * batch];
    int top_uids[K * batch], real_uids[K * batch];
    int target = 1123;
    for (int i = 0; i < batch; ++i)
        real_uids[i] = target + i;
    const int nIter = 5;
    gettimeofday(&time1, &zone);
    for (int i = 0; i < nIter; ++i)
        ret = probe->query(&x[target * Dimension], batch, ga, &sims[0], &top_uids[0]);
    gettimeofday(&time2, &zone);
    if (ret == 0){
        printf("Query Faieled\n");
        return -1;
    }
    float delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;

    float QPS = 1000 / (delta / nIter) * batch;

    printf("[%10s] <%15s> Batch [%3d] Query On %8d Gallery, cost:%4fms, QPS: %.2f\n", target_name.c_str(), , batch, N, delta/nIter, QPS);

    if (test_equal(sims, top_uids, real_uids, batch, K, true, true, dis_type) == false){
        return -1;
    }

    delete probe;
    delete ga;
    return 0;
}

TEST_F(UnitTest, QueryTest){
    EXPECT_EQ(0, (test_query<float, rsearch::COSINE>(30000, 128, 512, rsearch::DUMMY)) );
    EXPECT_EQ(0, (test_query<int8_t, rsearch::COSINE>(30000, 128, 512, rsearch::DUMMY)) );
    EXPECT_EQ(0, (test_query<float, rsearch::COSINE>(30000, 128, 512, rsearch::X86_RAPID)) );
    EXPECT_EQ(0, (test_query<int8_t, rsearch::COSINE>(30000, 128, 512, rsearch::X86_RAPID)) );
    EXPECT_EQ(0, (test_query<float, rsearch::COSINE>(30000, 128, 512, rsearch::X86_PQIVF)) );
    EXPECT_EQ(0, (test_query<int8_t, rsearch::COSINE>(30000, 128, 512, rsearch::X86_PQIVF)) );

}