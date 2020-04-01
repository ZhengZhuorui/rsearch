#include "utils/mt_manager.h"
namespace rsearch{
MthManager::MthManager(){
    this->nprocs = std::thread::hardware_concurrency();
    this->task_vec.clear();
    this->used.resize(this->nprocs);
    for (int i = 0; i < this->nprocs; ++i)
        this->used[i] = false;
}
MthManager::~MthManager(){
    this->task_vec.clear();
    this->used.clear();
}
void MthManager::add(task& t){
    std::lock_guard<std::mutex> lck(this->task_mutex);
    this->task_vec.push(t);
}
void MthManager::work(){
    while(true){
        int thread_id = -1;
        for (int i = 0; i < nprocs; ++i)
        if (this->used[i] == false){
            this->m_mutex.lock();
            if (this->used[i] == false){
                thread_id = i;
                this->used[i] = true;
                this->m_mutex.unlock();
                break;
            }
            this->m_mutex.unlock();
        }
        if (thread_id != -1){
            this->task_mutex.lock();
            task t = this->task_vec.front();
            this->task_vec.unlock();
            t(thread_id);
            return;
        }
    }
}
}