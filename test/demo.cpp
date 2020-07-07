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
rsearch::idx_t uids[10000];

int main(){ 
    rsearch::get_random_data<float, rsearch::EUCLIDEAN>(data, n, dimension);
    rsearch::probe<T> *p = rsearch::create_probe<T>(dimension, topk, rsearch::EUCLIDEAN, rsearch::X86_PQIVF);
    rsearch::gallery<T> *ga;
    p->create_gallery(&ga);
    ga->init();
    ga->add(data.data(), n);
    p->query(data.data() + 1000 * dimension, 10, ga, sims, uids);
}
