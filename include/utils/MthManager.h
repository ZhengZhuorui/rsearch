#pragma once
#include <bits/stdc++.h>
namespace rsearch{
class MthManager{
public:
    typedef std::function<void(int)> task;
    MthManager();
    ~MthManager();
    void work();
    void add_task(task&);
    int size();
private:
    //std::vector<bool> used;
    std::queue<task> task_vec;
    std::queue<int> used;

    std::condition_variable cv;
    int nprocs;
    std::mutex task_mutex;
    std::mutex used_mutex;
};

}