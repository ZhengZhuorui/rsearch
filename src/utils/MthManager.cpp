#include "utils/MthManager.h"
namespace rsearch{
MthManager::MthManager(){
    this->nprocs = std::thread::hardware_concurrency();
    //this->nprocs = 1;
    for (int i = 0; i < this->nprocs; ++i)
        this->used.push(i);
}
MthManager::~MthManager(){
    //this->task_vec.clear();
    while (this->task_vec.empty() == false)
        this->task_vec.pop();
    while (this->used.empty() == false)
        this->used.pop();
}
void MthManager::add_task(task& t){
    //std::lock_guard<std::mutex> lck(this->task_mutex);
    this->task_vec.push(t);
}

void MthManager::work(){
    int thread_id = -1;
    //this->used_mutex.lock();
    this->used_mutex.lock();
    while (thread_id == -1){
        if (this->used.empty() == false){
            thread_id = this->used.front();
            this->used.pop();
        }
    }
    task t = this->task_vec.front();
    this->task_vec.pop();
    this->used_mutex.unlock();

    t(thread_id);

    this->used.push(thread_id);
    
}
int MthManager::size(){
    //std::lock_guard<std::mutex> lck(this->m_mutex);
    return this->task_vec.size();
}
}