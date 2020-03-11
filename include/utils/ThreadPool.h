#pragma once
#include <bits/stdc++.h>
namespace rsearch
{
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

ThreadPool::ThreadPool():m_mutex(),
                        is_started(false),
                        nprocs(std::thread::hardware_concurrency()){}
ThreadPool::~ThreadPool(){
    if (is_started == true)
        this->stop();
}
void ThreadPool::start(){
    this->is_started = true;
    this->m_threads.reserve(this->nprocs);
    for (int i = 0; i < this->nprocs; ++i){
        this->m_threads.push_back(new std::thread(std::bind(&ThreadPool::thread_loop, this)));
    }
}
void ThreadPool::stop(){
    this->m_mutex.lock();
    this->is_started = false;
    for (auto it : this->m_threads){
        it->join();
        delete it;
    }
    this->m_threads.clear();
}
void ThreadPool::add_task(const task& t){
    this->m_mutex.lock();
    this->m_task.push(t);
}

void ThreadPool::thread_loop(){
    while (this->is_started){
        task t = this->m_task.front();
        t();
    }
}



} // namespace rsearch


