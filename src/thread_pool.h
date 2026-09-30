// thread_pool.h — minimal fixed-size thread pool (std::thread only).
#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace haidass {

class ThreadPool {
public:
    // n_threads = total number of workers INCLUDING the calling thread's share.
    explicit ThreadPool(int n_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Splits [0, n) into ranges and runs fn(start, end) on workers + caller.
    void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn);

    int num_threads() const { return n_threads_; }

private:
    void worker_loop();

    int n_threads_ = 1;
    std::vector<std::thread> workers_;

    std::mutex mtx_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    const std::function<void(int64_t, int64_t)>* fn_ = nullptr;
    int64_t n_ = 0;
    int64_t next_ = 0;        // next chunk start (dynamic scheduling)
    int64_t chunk_ = 1;
    int generation_ = 0;
    int active_ = 0;
    bool stop_ = false;
};

} // namespace haidass
