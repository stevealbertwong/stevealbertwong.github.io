---
layout: post
comments: true
title:  "Classic Dining Philosophers deadlock problem, solutions and its analogy to deadlock in multi-threaded server"
excerpt: "Classic Dining Philosophers deadlock problem, solving it using condition variable and semaphore and its analogy to deadlock in multi-threaded server"
date:   2017-01-20 11:00:00
mathjax: true
---



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

Solution:

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

