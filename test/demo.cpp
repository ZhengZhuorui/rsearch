#include "rsearch_type.h"
#include "utils/utils.h"
#include "probe/cpu_base_probe.h"
#include "gallery/cpu_base_gallery.h"

#include <bits/stdc++.h>
using namespace std;

int n = 10000;
int dimension = 512;
int topk = 128;
float data[5120000], sims[10000];
uint32_t uids[10000];

int main(){ 
    rsearch::get_random_data<float>(data, n, dimension);
    rsearch::norm(data, n, dimension);
    printf("target 1, n=%d, dimension=%d\n", n, dimension);
    for (int i = 0; i < dimension; ++i)
        printf("%.4f ", data[1000 *dimension + i]);
    printf("\n");
    rsearch::probe<float, rsearch::COSINE> *p = rsearch::create_probe<float, rsearch::COSINE>(dimension, topk, rsearch::X86_RAPID);
    rsearch::gallery<float, rsearch::COSINE> *ga;
    
    printf("target 2\n");
    p->create_gallery(&ga);
    ga->add(data, n);
    ga->init();
    //printf("%u\n", ((rsearch::cpu_base_gallery<float, rsearch::COSINE>*)ga)->num);
    printf("target 3\n");
    p->query(data + 1000 * dimension, 10, ga, sims, uids);
    for (int i = 0; i < 10; ++i){
        printf("%f %u\n", sims[i * topk], uids[i * topk]);
    }
    
}
