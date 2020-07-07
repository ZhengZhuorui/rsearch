#include "utils/cluster.h"
#include "probe/rsearch_probe.h"
#include "gallery/rsearch_gallery.h"
#include <sys/time.h>
namespace rsearch{
using std::pair;
template<typename T,
        DistanceType dist_type>
void k_means_parallel(int thread_id, matrix_mul<T>** mm, const T* data, const T* res_tmp, const T* res_offset, T* res, int* cluster_num, int cluster_center,
                     int size, int dimension, volatile T* tot_dis){
    pair<T, idx_t>* dis;
    mm[thread_id]->mul(data, res_tmp, res_offset, size, cluster_center, &dis);
    T tv = 0;
    for (int k = 0; k < size; ++k){
        cluster_num[dis[k].second]++;
        for (int l = 0; l < dimension; ++l){
            res[dis[k].second * dimension + l] += data[1LL * k * dimension + l];
        }
        T v = vec_dis<T, dist_type>(data + 1LL * k * dimension, res_tmp + dis[k].second * dimension, dimension);
        tv += v;
    }
    (*tot_dis) += tv;
}

template<typename T,
        DistanceType dist_type>
int k_means(const T* data, const int n, const int cluster_center, const int dimension, std::vector<T>& res){
    //using Tout = typemap_t<T>;
    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    #ifdef DEBUG_KMEANS
    std::cout << "[k_means] begin, cluster = " << cluster_center << ", dimension = "<< dimension << std::endl;
    //for (int i = 0; i < dimension; ++i)
    //    std::cout << (float)data[i] << "?";
    //std::cout << std::endl;
    #endif
    //srand(time(NULL));
    res.resize(cluster_center * dimension);
    vector<int> cluster_num(cluster_center);
    vector<T> res_offset(cluster_center);
    vector<T> res_tmp(cluster_center * dimension);
    memset(res.data(), 0, cluster_center * dimension * sizeof(T));
    memset(res_tmp.data(), 0, cluster_center * dimension * sizeof(T));
    memset(cluster_num.data(), 0, cluster_center * sizeof(int));
    for (int i = 0; i < cluster_center; ++i){
        int id = rand() % n;
        #ifdef DEBUG_KMEANS
            //printf("%d ",id);
        #endif
        memcpy(res.data() + 1LL * i * dimension, data + 1LL * id * dimension, dimension * sizeof(T));
        res_offset[i] = get_offset<T, dist_type>(res.data() + i * dimension, dimension);
    }
    
    //int nprocs = std::thread::hardware_concurrency();
    vector<matrix_mul<T>*> mm(12);
    int max_batch = 32;
    for (int i = 0; i < 12; ++i){
        mm[i] = new rapid_matrix_mul<T>;
        mm[i]->set(dimension, 1, max_batch, cluster_center);
    }
    //mm = new rapid_matrix_mul<T>;
    //mm->set(dimension, 1, max_batch, cluster_center);
    int i, flag;
    //pair<T, idx_t>* dis;
    vector<gallery<T>*> ga(10);
    vector<probe<T>*> p(10);
    vector<int> st(10);
    /*for (int i = 0; i < 8 ; ++i){
        p[i] = create_probe<T>(dimension, 1, dist_type, X86_RAPID);
        p[i]->create_gallery(&ga[i]);
        ga[i]->init();
        ga[i]->add(res.data(), cluster_center);
        st[i] = n / 8 * i;
    }
    st[8] = n;
    T sims[n];
    idx_t uids[n];*/
    ThreadPool* threadpool = new ThreadPool;
    MthManager* mth_manager = new MthManager;
    threadpool->start();
    float delta = 0;
    for (i = 0; i < 15; ++i){

        gettimeofday(&time1, &zone);
        #ifdef DEBUG_KMEANS
        T tot_dis = 0;
        std::cout<< "[kmeans] cycle = " << i << std::endl;
        #endif
        memset(cluster_num.data(), 0, 1LL * cluster_center * sizeof(int));
        memcpy(res_tmp.data(), res.data(), 1LL * cluster_center * dimension * sizeof(T));
        memset(res.data(), 0, 1LL * cluster_center * dimension * sizeof(T));
        /*
        #pragma omp parallel for
        for (int j = 0; j < 8; ++j){
            printf("query j=%d, st[j]= %d, st[j+1]=%d\n",j, st[j], st[j+1]);
            p[j]->query(data + st[j], st[j+1] - st[j], ga[j], sims + st[j], uids + st[j]);
        }
        printf("query end.\n");
        //p->query(data, n, ga, sims, uids);
        for (int j = 0; j < n; ++j){
            //std::cout << uids[j] << std::endl;
            cluster_num[uids[j]]++;
            for (int l = 0; l < dimension; ++l){
                res[uids[j] * dimension + l] += data[j * dimension + l];
            }
            #ifdef DEBUG_KMEANS
            //T v = vec_dis<T, dist_type>(data + uids[j] * dimension, res_tmp.data() + dis[k].second * dimension, dimension);
            tot_dis += sims[j];
            //std::cout << tot_dis << " " << v  << std::endl;
            #endif
        }*/
        /*
        for (int j = 0; j < n; j+=max_batch){
            int size = std::min(max_batch, n - j);
            int id = (j/max_batch)%12;
            pair<T, idx_t>* dis;
            mm[id]->mul(data + j * dimension, res_tmp.data(), res_offset.data(), size, cluster_center, &dis);
            //std::cout<< "[kmeans] ? " << std::endl;
            for (int k = 0; k < size; ++k){
                cluster_num[dis[k].second]++;
                for (int l = 0; l < dimension; ++l){
                    res[dis[k].second * dimension + l] += data[(j + k) * dimension + l];
                }
                #ifdef DEBUG_KMEANS
                T v = vec_dis<T, dist_type>(data + (j + k) * dimension, res_tmp.data() + dis[k].second * dimension, dimension);
                tot_dis += v;
                //std::cout << tot_dis << " " << v  << std::endl;
                #endif
            }
        }*/
        for (int j = 0; j < n; j+=max_batch){
            int size = std::min(max_batch, n - j);
            std::function<void(int)> f = std::bind(k_means_parallel<T, dist_type>, std::placeholders::_1, mm.data(), data + 1LL * j * dimension, res_tmp.data(),
             res_offset.data(), res.data(), cluster_num.data(), cluster_center, size, dimension, &tot_dis);
            mth_manager->add_task(f);
        }
        std::function<void()> work = std::bind(&MthManager::work, mth_manager);
        int work_sz = mth_manager->size();
        //std::cout << work_sz << std::endl;
        for (int j = 0; j < work_sz; ++j)
        threadpool->add_task(work);

        threadpool->synchronize();

        std::cout << "tot dis = " << tot_dis << std::endl;
        flag = 0;

        #pragma omp parallel for
        for (int j = 0; j < cluster_center; ++j){
            if (cluster_num[j] == 0){
                #ifdef DEBUG_KMEANS
                    //std::cout<< " random " << " ";
                #endif
                int id = rand() % n;
                memcpy(res.data() + j * dimension, data + 1LL * id * dimension, dimension * sizeof(T));
            }
            else{
                for (int k = 0; k < dimension; ++k){
                    res[j * dimension + k] /= cluster_num[j];
                }
            }
        }
        if (dist_type == rsearch::COSINE)
            norm(res.data(), cluster_center, dimension);
            
        std::cout << "tot dis = " << tot_dis << std::endl;
        //#pragma omp parallel for
        for (int j = 0; j < cluster_center; ++j)
            res_offset[j] = get_offset<T, dist_type>(res.data() + j * dimension, dimension);
        #ifdef DEBUG_KMEANS
        
        
        #endif
        for (int j = 0; j < cluster_center * dimension; ++j) 
        if (res[j] != res_tmp[j]){
            flag = 1;
            break;
        }
        /*
        for (int j = 0 ; j < 8; ++j){
            ga[j]->reset();
            ga[j]->add(res.data(), cluster_center);
        }
        */
        gettimeofday(&time2, &zone);
        delta += (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
        std::cout << "time:" << delta << std::endl;
        if (flag == 0) break;
    }
    threadpool->stop();
    delete threadpool;
    delete mth_manager;
    /*
    for (int j = 0 ; j < 8; ++j){
        delete ga[j];
        delete p[j];
    }*/
    printf("[kmeans]train cycle: %d\n", i);
    return 0;
}
template int k_means<float, COSINE>(const float* data, const int n, const int cluster_center, const int dimension, std::vector<float>& res);
template int k_means<float, EUCLIDEAN>(const float* data, const int n, const int cluster_center, const int dimension, std::vector<float>& res);


}
