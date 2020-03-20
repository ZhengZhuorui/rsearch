#include "rsearch_def.h"
#include "probe/rsearch_probe.h"
#include "gallery/rsearch_gallery.h"
#include "utils/utils.h"

#include <bits/stdc++.h>
using namespace std;
using T = float;
using Tout = rsearch::typemap_t<T>;
int n = 10000;
int dimension = 512;
int topk = 128;
float sims[10000];
vector<T> data;
uint32_t uids[10000];

int main(){ 
    rsearch::get_random_data<float, rsearch::EUCLIDEAN>(data, n, dimension);
    printf("target 1, n=%d, dimension=%d\n", n, dimension);
    for (int i = 0; i < dimension; ++i)
        printf("%.4f ", data[1000 *dimension + i]);
    printf("\n");
    rsearch::probe<T, rsearch::EUCLIDEAN> *p = rsearch::create_probe<T, rsearch::EUCLIDEAN>(dimension, topk, rsearch::X86_RAPID);
    rsearch::gallery<T, rsearch::EUCLIDEAN> *ga;
    
    printf("target 2\n");
    p->create_gallery(&ga);
    ga->add(data.data(), n);
    ga->init();
    printf("target 3\n");
    p->query(data.data() + 1000 * dimension, 10, ga, sims, uids);
    for (int i = 0; i < 10; ++i){
        printf("%f %u\n", sims[i * topk], uids[i * topk]);
    }
    
}
