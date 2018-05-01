---
layout: post
comments: true
title:  "Classic Dining Philosophers deadlock problem in multi-threaded server, threadpool with load balancing in scaling modern web architecture distributed system."
excerpt: "Classic Dining Philosophers deadlock problem is solvable using condition variable, semaphore and threadpool. Such solution goes beyond just solving deadlock in multi-threaded server. It also provides an introduction to understand load balancing and threadpoll in scaling distribution system in modern web architecture."
date:   2017-01-20 11:00:00
mathjax: true
---

## Classic Dining Philosophers to demonstrate Deadlock 

![](/assets/miscellaneous/dining-philosopher.png)

The key to understanding deadlock in multi-threading is an analogy drawn to the deadlock happened in dining philosophers. If there were 5 philosophers and 5 forks, and since all of them being philosophers are egocentric, they all share same narcissistic behavior to take left fork first then right fork both before eating, and they won't let go of left fork if they cannot grab the right fork even though it means other people won't be able to eat. 

This type of philosophical behavior will result in a very likely situation where every one of them grab left fork and so there were no right fork available for any of them. Since they won't let go of left fork and so it resulted in a deadlock situation where no one is able to grab both fork and eat.

While this might not happen in real life dining table, it is a PERFECT ANALOGY to the design of threads and locks that could cause your multi-threaded server with locks to synchronize resources access to stop running.

Try running the following code, if you run it many times, there will be situation the code just stop running
```
/*
author: steven wong

g++ -std=c++11 dining-philosophers-deadlock.cc -o dining-philosophers-deadlock

*/
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <cstdlib> 
#include <chrono>
#include <unistd.h>
#include <condition_variable>

using namespace std;
static const unsigned int kNumPhilosophers = 5;
static const unsigned int kNumForks = kNumPhilosophers;
static const unsigned int kNumMeals = 1;

static mutex forks[kNumForks];
static mutex numThreadsAllowedLock;
static mutex streamLock;

// simulate threads starting at different time
void randomizeStartTime(int num){
    std::this_thread::sleep_for(std::chrono::milliseconds(rand()%2000+100));
    // sleep(rand()%3);
        
    streamLock.lock();
    cout << " i am thinking : from thread" << num << endl;
    streamLock.unlock();
}

void eat(int num){  
    int left = num;
    int right = (num+1) % kNumPhilosophers; 

    // START OF CRITICAL REGION
    forks[left].lock();
    // change 900 to 100 to make deadlock happend half of the time
    std::this_thread::sleep_for(std::chrono::milliseconds(rand()%1000+900));
    forks[right].lock();
    
    streamLock.lock();
    cout << " starts eating : from thread" << num << endl;
    streamLock.unlock();
    
    // sleep(1);
    
    streamLock.lock();
    cout << " finishes eating : from thread" << num << endl;
    streamLock.unlock();    


    forks[left].unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(rand()%1000+100));
    forks[right].unlock();  
    // END OF CRITICAL REGION
}

void philosopher(int id){
    for (int i = 0; i < kNumMeals; ++i)
    {
        randomizeStartTime(id);
        eat(id); 
    }
}

int main(int argc, char const *argv[])
{
    std::vector<std::thread> threads;       
    for (int i = 0; i < kNumPhilosophers; i++){
        threads.push_back(thread(philosopher, i));
    }
    
    for (auto& p: threads) {
        p.join();
    }
    
    return 0;
}
```

The reason it stops running is because the lock is designed in a way that there are multiple locks inside locks in deadlock sequence. All the thread could go through 1st lock, waiting for the 2nd lock which is already gone through by another thread.

### Solution:

Other than design the locks sequence, logically we could limit the threads to only 4 locks so at least 1 thread could go through both locks and release both locks.

We could achieve it with busy waiting, condition variable and semaphore. They all are very similar in terms of limiting the amount of threads going through critical region. Condition variable could send and receive signal so it won't hoards cpu time like busy waiting by constanting checking numThreadsAllowed in critical region. Semaphore is condition variable with condition numThreadsAllowed.

```
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <cstdlib> 
#include <chrono>
#include <unistd.h>
#include <condition_variable>

using namespace std;
static const unsigned int kNumPhilosophers = 3;
static const unsigned int kNumForks = kNumPhilosophers;
static const unsigned int kNumMeals = 1;

static mutex forks[kNumForks];
static mutex numThreadsAllowedLock;
static mutex streamLock;

// KEY IDEA OF BUSY WAITING !!!
static unsigned int numThreadsAllowed = kNumForks-1;
static void busyWaiting(){
    while(true){
        numThreadsAllowedLock.lock();
        if (numThreadsAllowed > 0) // if still quota in critical region
        {           
            numThreadsAllowed --;           
            break;
        }
        numThreadsAllowedLock.unlock();

        streamLock.lock();
        cout << "no. threads in critical region is full, let me sleep then check again and hoards cpu time" << endl;
        streamLock.unlock();                        
        sleep(1);
    }
    numThreadsAllowedLock.unlock();
}

static void busyWaitingPermission(){
    numThreadsAllowedLock.lock();
    numThreadsAllowed++;
    numThreadsAllowedLock.unlock();
}

// KEY IDEA OF CONDITION VARIABLE !!!
static mutex cvLock;
static condition_variable cv;

static void cvWaiting(){
    unique_lock<mutex> ul(cvLock); // automates locking of mutex
    while (numThreadsAllowed == 0){
        cv.wait(ul);
    }   
    numThreadsAllowed--;
}

static void cvWaitingPermission() {
  unique_lock<mutex> ul(cvLock);
  numThreadsAllowed++;
  cv.notify_one();  
}

// ALTERNATIVELY USE CONDITION VARIABLE ANY
static mutex cvaLock;
static condition_variable_any cva;
static void cvaWaiting(){
    lock_guard<mutex> lg(cvaLock); // automates locking of mutex
    // lambda 2nd arg: condition of cv
    cva.wait(cvaLock, []{ return numThreadsAllowed > 0;});  
    numThreadsAllowed--;
}

static void cvaWaitingPermission() {
  lock_guard<mutex> lg(cvaLock);
  numThreadsAllowed++;
  if (numThreadsAllowed == 1) cva.notify_all();
}

// simulate threads starting at different time
void randomizeStartTime(int num){
    std::this_thread::sleep_for(std::chrono::milliseconds(rand()%2000+100));
    // sleep(rand()%3);
        
    streamLock.lock();
    cout << " i am thinking : from thread" << num << endl;
    streamLock.unlock();
}

void eat(int num){  
    int left = num;
    int right = (num+1) % kNumPhilosophers; 

    // busyWaiting();
    // cvWaiting();
    cvaWaiting();


    // CRITICAL REGION
    forks[left].lock();
    // change 900 to 100 to make deadlock happend half of the time
    std::this_thread::sleep_for(std::chrono::milliseconds(rand()%1000+900));
    forks[right].lock();
    
    streamLock.lock();
    cout << " starts eating : from thread" << num << endl;
    streamLock.unlock();
    
    // sleep(1);
    
    streamLock.lock();
    cout << " finishes eating : from thread" << num << endl;
    streamLock.unlock();    

    
    // cvWaitingPermission();
    cvaWaitingPermission();
    // busyWaitingPermission(); 


    forks[left].unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(rand()%1000+100));
    forks[right].unlock();

    // busyWaitingPermission(); // if put here, need to wait for other threads to unlock

}

void philosopher(int id){
    for (int i = 0; i < kNumMeals; ++i)
    {
        randomizeStartTime(id);
        eat(id); 
    }
}

int main(int argc, char const *argv[])
{
    std::vector<std::thread> threads;       
    for (int i = 0; i < kNumPhilosophers; i++){
        threads.push_back(thread(philosopher, i));
    }
    
    for (auto& p: threads) {
        p.join();
    }
    
    return 0;
}
```

The above code actually resembles a multi-threaded server with locks to synchronize resources access. Let me rename the code a little bit. Think of philosopher as threaded function to deal with logic when received a request. Depending on the request types (e.g. post, get, proxy, other headers fields, protocol version etc), resources synchronization lock (e.g. database access, cache, blacklist etc.) could result in a deadlock manner when request comes in parallel fashion.


```
/*
g++ -std=c++11 server-deadlock.cc -o server-deadlock
*/
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <cstdlib> 
#include <chrono>
#include <unistd.h>
#include <condition_variable>

using namespace std;
static const unsigned int kNumTypesRequests = 5;
static const unsigned int kNumResourceLocks = kNumTypesRequests;
static const unsigned int kNumRequests = 1;

static mutex resourceLocks[kNumResourceLocks];
static mutex numThreadsAllowedLock;
static mutex streamLock;

// KEY IDEA OF BUSY WAITING !!!
static unsigned int numThreadsAllowed = kNumResourceLocks-1;
static void busyWaiting(){
    while(true){
        numThreadsAllowedLock.lock();
        if (numThreadsAllowed > 0) // if still quota in critical region
        {           
            numThreadsAllowed --;           
            break;
        }
        numThreadsAllowedLock.unlock();

        streamLock.lock();
        cout << "no. threads in critical region is full, let me sleep then check again and hoards cpu time" << endl;
        streamLock.unlock();                        
        sleep(1);
    }
    numThreadsAllowedLock.unlock();
}

static void busyWaitingPermission(){
    numThreadsAllowedLock.lock();
    numThreadsAllowed++;
    numThreadsAllowedLock.unlock();
}

// KEY IDEA OF CONDITION VARIABLE !!!
static mutex cvLock;
static condition_variable cv;

static void cvWaiting(){
    unique_lock<mutex> ul(cvLock); // automates locking of mutex
    while (numThreadsAllowed == 0){
        cv.wait(ul);
    }   
    numThreadsAllowed--;
}

static void cvWaitingPermission() {
  unique_lock<mutex> ul(cvLock);
  numThreadsAllowed++;
  cv.notify_one();  
}

// ALTERNATIVELY USE CONDITION VARIABLE ANY
static mutex cvaLock;
static condition_variable_any cva;
static void cvaWaiting(){
    lock_guard<mutex> lg(cvaLock); // automates locking of mutex
    // lambda 2nd arg: condition of cv
    cva.wait(cvaLock, []{ return numThreadsAllowed > 0;});  
    numThreadsAllowed--;
}

static void cvaWaitingPermission() {
  lock_guard<mutex> lg(cvaLock);
  numThreadsAllowed++;
  if (numThreadsAllowed == 1) cva.notify_all();
}

// simulate threads starting at different time
void randomizeStartTime(int num){
    std::this_thread::sleep_for(std::chrono::milliseconds(rand()%2000+100));
    // sleep(rand()%3);
        
    streamLock.lock();
    cout << " i am thinking : from thread" << num << endl;
    streamLock.unlock();
}

void replyRequest(int num){ 
    int left = num;
    int right = (num+1) % kNumTypesRequests;    

    // busyWaiting();
    // cvWaiting();
    cvaWaiting();


    // CRITICAL REGION
    resourceLocks[left].lock();
    // change 900 to 100 to make deadlock happend half of the time
    std::this_thread::sleep_for(std::chrono::milliseconds(rand()%1000+900));
    resourceLocks[right].lock();
    
    streamLock.lock();
    cout << " starts resource access : from thread" << num << endl;
    streamLock.unlock();
    
    // sleep(1);
    
    streamLock.lock();
    cout << " finishes resource access : from thread" << num << endl;
    streamLock.unlock();    

    
    // cvWaitingPermission();
    cvaWaitingPermission();
    // busyWaitingPermission(); 


    resourceLocks[left].unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(rand()%1000+100));
    resourceLocks[right].unlock();

    // busyWaitingPermission(); // if put here, need to wait for other threads to unlock

}

void serverDealWithRequest(int id){
    for (int i = 0; i < kNumRequests; ++i)
    {
        randomizeStartTime(id);
        replyRequest(id); 
    }
}

int main(int argc, char const *argv[])
{
    std::vector<std::thread> threads;       
    for (int i = 0; i < kNumTypesRequests; i++){
        threads.push_back(thread(serverDealWithRequest, i));
    }
    
    for (auto& p: threads) {
        p.join();
    }
    
    return 0;
}
```

## Threadpool

One of the direct solution to dining philospher problem is limit the number of threads. Threadpool is a pool of threads that will not die. These threads will wait diligently after finish task for another new task until they are told to stop. Threadpool reduces the overhead of creating thread as such operation is expensive and creates persistent connection between processes across distribution system. For instance, Facebook uses threadpool in web server and udp packets to creates persistent connection with memcache servers to serve 90% of its content without querying its database. 


```
/*
Author: Steven Wong

worker thread (operator())
1. cv.wait() + predicate for each thread
2. move + pop
3. task()


dispatcher thread (enqueue())
1. packaged_task
2. get_future()
3. lock_guard + emplace
4. cv.notify()

g++ -std=c++11 -g -O0 main.cpp -o main
*/
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <iostream>

#define LOG(x) std::cout << x << std::endl;

class ThreadPool
{
public:
    ThreadPool(int threads) : stop(false), numThreads(threads), workers(std::vector<std::thread>(threads)) {this->init();};    
    
    template<typename F, typename...Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;


    template<typename F>
    auto enqueue(F && f) 
        ->std::future<decltype(f(0))>;
    
    ~ThreadPool();

private:
    void init();
    bool stop; 
    size_t numThreads;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::vector< std::thread > workers;
    std::queue< std::function<void()> > tasks;

    // functor => web server logic e.g. memcached, DB access(replicate, partition, specialize)
    class ThreadWorker {
    private:
      int tid;      
      ThreadPool * tpool; // nested class access same parent(reference) class's variable
    public:      
      ThreadWorker(ThreadPool * pool, const int id)
        : tpool(pool), tid(id) {}

      void operator()() {        
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(tpool->queue_mutex);
            tpool->cv.wait(lock, [this]{return this->tpool->stop || !this->tpool->tasks.empty();});
            if(tpool->stop && tpool->tasks.empty())
                return;
            task = std::move(tpool->tasks.front());
            tpool->tasks.pop();            
          }
          task();
          }
      }   
    };
};

inline void ThreadPool::init(){
    // naive load balancing
    for (int i = 0; i < workers.size(); ++i) {        
      workers[i] = std::thread(ThreadPool::ThreadWorker(this, i));
    }
}


// with arguments
template<typename F, typename...Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> // TODO return type??
{
    using return_type = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared< std::packaged_task<return_type()> >(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");                
        tasks.emplace([task](){ (*task)(); });
    }
    cv.notify_one();
    return res;
}

// function overload without arguments
template<typename F>
auto ThreadPool::enqueue(F && f) 
    ->std::future<decltype(f(0))> 
{    
    auto task = std::make_shared<std::packaged_task<decltype(f(0))(int)>>(std::forward<F>(f));
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");                
        tasks.emplace([task](){ (*task)(); });
    }
    cv.notify_one();
    return task->get_future();
}

inline ThreadPool::~ThreadPool(){
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    cv.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}
#endif

```

[source on github](https://github.com/stevealbertwong/threadpool/tree/master/steven-threadpool)

A couple of function might be newer as it is introduced in c++11. std::move() converts lvalue (variable with address) to rvalue (temporary object without address). Rvalue reference is introduced to avoid the performance hit due to deep copy and convenience. Here std::move() is used since std::function<void> is rvalue. For instance, std::emplace_back: appends rvalue reference to the queue and call thread() on the fly so there is no unnecessary copy.

std::future and std::packaged_task together provides a way to to block parent thread and wait until child thread return. std::packaged_task could act as an asychronous callable wrapper that won't run until task is invoked.

Thread pool plays an important part in modern web distributed system architecture. In particular load balancing and web servers.

![](https://raw.githubusercontent.com/stevealbertwong/threadpool/master/web_architecture.png)

See the above picture and following code. void ThreadPool::load_balance() optimizes to not spawning all threads right away until threadpool is fully occupied. It maintains a data strucutre of thread workers to assign work to free thread void ThreadPool::worker(size_t id).

One could easily add logic to check complexity of task, report thread's progress, prioritize function, and combine different threads to work on one task. The idea is that synchronization, threadpool and data structure play a essential part in understanding distributed system in modern web architecture.

## Threadpool with load balancing

```
#include "thread-pool.h"

using namespace std;

ThreadPool::ThreadPool(size_t numThreads) { 
    workers = vector<worker_t>(numThreads);

    max_allowed_sema.reset(new semaphore(numThreads));  
    tasks_sema.reset(new semaphore(0));
    wait_sema.reset(new semaphore(0));

    num_active_threads = 0;
    tasks_done = 0;


    for (size_t workerID = 0; workerID < numThreads; workerID++) {
        workers[workerID].ready_to_exec.reset(new semaphore(0));        
        workers[workerID].thunk = NULL;
    }   

    thread dt([this]() -> void { 
        this->load_balance();
    });
    dt.detach();    
}

// surgery code to join threads
void ThreadPool::wait(){
    wait_sema->wait();  
}

void ThreadPool::enqueue(const std::function<void(void)>& thunk) {  
    tasks_done++;
    tasks_lock.lock();
    tasks.push(thunk);
    tasks_lock.unlock();
            
    tasks_sema->signal();
}

void ThreadPool::load_balance() {
    while (true) {

        // wait for function attached
        tasks_sema->wait();     
        // max threads allowed(loop to get threads) 
        max_allowed_sema->wait();       

        // if no free thread, spawn new worker thread
        if(free_threads.empty()){           
            
            tasks_lock.lock(); // protect tasks, no enqueue when send to thread
            workers[num_active_threads].thunk = tasks.front();
            tasks.pop();            
            tasks_lock.unlock();

            std::thread wt([this](size_t num_active_threads) -> void {
                this->worker(num_active_threads);
            }, num_active_threads); // std:bind ??
            wt.detach();
            
            workers[num_active_threads].ready_to_exec->signal();
            num_active_threads++; 

        // if yes existing thread
        }else{
            free_threads_lock.lock();
            int id = free_threads.front();
            free_threads.pop();
            free_threads_lock.unlock();

            tasks_lock.lock();          
            workers[id].thunk = tasks.front();  
            tasks.pop();        
            tasks_lock.unlock();                        
            workers[id].ready_to_exec->signal();
        }       
    }
}


/* 
THREAD WILL NOT DIE, IT WILL JUST WAIT TO BE REUSED AFTER ONE LOOP
MULTI-THREADED FUNCTION

worker == web server logic(thunk())
its predicator == load_balance's signal
*/
void ThreadPool::worker(size_t id) {
    while (true) {          
        workers[id].ready_to_exec->wait();
                        
        /* NOT LOCKED!!!!! OTHERWISE NOT MULTI-THREADING */     
        workers[id].thunk();
        workers[id].thunk = NULL;
        
        free_threads_lock.lock();
        free_threads.push(id);
        free_threads_lock.unlock();

        max_allowed_sema->signal(); 
        
        tasks_done--;
        if(tasks_done == 0){
            wait_sema->signal();
        }
    }
}
```

[source on github](https://github.com/stevealbertwong/threadpool/tree/master/loadbalance-threadpool)

reference:

[CMU 15418 parallel programming: scaling website](http://15418.courses.cs.cmu.edu/spring2016content/lectures/14_webscaling/14_webscaling_slides.pdf)

[std::packaged_task examples](http://thispointer.com/c11-multithreading-part-10-packaged_task-example-and-tutorial/)

[std::packaged_task examples 2](http://www.modernescpp.com/index.php/asynchronous-callable-wrappers)
