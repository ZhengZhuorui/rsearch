#include "utils/MthManager.h"
namespace rsearch{
MthManager::MthManager(){
    this->nprocs = std::thread::hardware_concurrency();
    //this->nprocs = 1;
    this->used.resize(this->nprocs);
    for (int i = 0; i < this->nprocs; ++i)
        this->used[i] = false;
}
MthManager::~MthManager(){
    //this->task_vec.clear();
    while (this->task_vec.empty() == false)
        this->task_vec.pop();
    this->used.clear();
}
void MthManager::add_task(task& t){
    std::lock_guard<std::mutex> lck(this->task_mutex);
    this->task_vec.push(t);
}
void MthManager::work(){
    if (this->task_vec.size() == 0)
        return;
    while (true){
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
        //std::cout << "[work]" << thread_id << std::endl;
        if (thread_id != -1){
            
            this->task_mutex.lock();
            task t = this->task_vec.front();
            this->task_vec.pop();
            this->task_mutex.unlock();

            t(thread_id);

            this->used[thread_id] = false;
            
            return;
        }
    }
}
int MthManager::size(){
    std::lock_guard<std::mutex> lck(this->m_mutex);
    return this->task_vec.size();
}
}