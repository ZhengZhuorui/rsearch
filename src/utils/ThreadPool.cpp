#include "utils/ThreadPool.h"
namespace rsearch{
ThreadPool::ThreadPool():m_mutex(),
                        is_started(false){
    this->nprocs = std::thread::hardware_concurrency();
    this->un_work_num = this->nprocs;
}
ThreadPool::~ThreadPool(){
    if (is_started == true)
        this->stop();
}
void ThreadPool::start(){
    this->is_started = true;
    for (int i = 0; i < this->nprocs; ++i){
        this->m_threads.push_back(new std::thread(std::bind(&ThreadPool::thread_loop, this)));
    }
}
void ThreadPool::synchronize(){
    std::unique_lock<std::mutex> lck(this->work_mutex);
    this->cv.wait(lck);
}

void ThreadPool::stop(){
    //std::lock_guard<std::mutex> lck(this->m_mutex);
    this->is_started = false;
    for (auto it : this->m_threads){
        if (it->joinable() == true)
            it->join();
        delete it;
    }
    this->m_threads.clear();
}
void ThreadPool::add_task(const task& t){
    std::lock_guard<std::mutex> lck(this->m_mutex);
    this->m_task.push(t);
}

void ThreadPool::thread_loop(){
    while (this->is_started == true){
        while (this->m_task.empty() == true && this->is_started == true){
            std::this_thread::yield();
        }
        task t = NULL;
        this->m_mutex.lock();
        if (this->m_task.empty() == false){
            --this->un_work_num;
            t = this->m_task.front();
            this->m_task.pop();
        }
        this->m_mutex.unlock();

        if (t != NULL){
            t();
            ++this->un_work_num;
        }
        if (this->un_work_num == this->nprocs && this->m_task.empty() == true){
            this->cv.notify_one();
        }

    }
    return;
}
}