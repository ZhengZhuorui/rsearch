#include "rsearch.h"
#include "probe/cpu_base_probe.h"
#include "gallery/cpu_base_gallery.h"
#include "gallery/cpu_rapid_gallery.h"
#include "utils/utils.h"
#include <bits/stdc++.h>
using namespace std;
using namespace rsearch;
int n = 10000;
int dimension = 512;
int topk = 128;
float data[5120000], sims[10000];
rsearch::uint64_t uids[10000];
void create_data(){
    ofstream fout;
    fout.open("/home/zzr/data/consine_data.exp", ofstream::binary);
    //float a = sqrt(1 / dimension);
    const int MO = 65535;
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < dimension; ++j){
            data[i * dimension + j] = 1.0 * (rand() % MO) / MO;
        }
    }
    r_bytes2file<float>(fout, data, n, dimension);
}
int main(){
    srand(time(NULL));
    
    ifstream fin;
    fin.open("/home/zzr/data/consine_data.exp", ifstream::binary);
    if (fin.is_open() == false){
        fin.close();
        create_data();
        fin.open("/home/zzr/data/consine_data.exp", ifstream::binary);
    }
    printf("read file\n");
    r_file2bytes<float>(fin, data, n, dimension);
    regular(data, n, dimension);
    printf("target 1, n=%d, dimension=%d\n", n, dimension);
    for (int i = 0; i < dimension; ++i)
        printf("%.4f ", data[1000 *dimension + i]);
    printf("\n");
    probe<float> *p = create_probe<float>(dimension, topk, X86_RAPID, EUCLIDEAN);
    gallery<float> *ga;
    
    printf("target 2\n");
    p->create_gallery(&ga);
    ga->add(data, n);
    ga->init();
    printf("%u\n", ((cpu_rapid_gallery<float>*)ga)->num);
    printf("target 3\n");
    p->query(data + 1000 * dimension, 10, ga, sims, uids);
    for (int i = 0; i < 10; ++i){
        printf("%f %llu\n", sims[i * topk], uids[i * topk]);
    }
    
}
