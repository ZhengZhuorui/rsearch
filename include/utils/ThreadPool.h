#pragma once
#include <bits/stdc++.h>
namespace rsearch{
class ThreadPool{
public:
    ThreadPool();
    ~ThreadPool();
    
    typedef std::function<int()> task;
    typedef std::queue<task> tasks;
    void start();
    void stop();
    void add_task(const task&);


private:
    ThreadPool(const ThreadPool&);
    const ThreadPool& operator=(const ThreadPool&);
    void thread_loop();
    

    std::vector<std::thread*> m_threads;
    std::mutex m_mutex;
    tasks m_task;
    int ThreadSize;
    bool is_started;
    int nprocs;
};

} // namespace rsearch


