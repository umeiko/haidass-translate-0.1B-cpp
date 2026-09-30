// thread_pool.cpp
#include "thread_pool.h"

#include <algorithm>

namespace haidass {

ThreadPool::ThreadPool(int n_threads) {
    n_threads_ = std::max(1, n_threads);
    for (int i = 1; i < n_threads_; ++i) workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
        ++generation_;
    }
    cv_start_.notify_all();
    for (auto& w : workers_) w.join();
}

void ThreadPool::parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    if (n_threads_ <= 1 || n <= 0) {
        if (n > 0) fn(0, n);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mtx_);
        fn_ = &fn;
        n_ = n;
        next_ = 0;
        // aim for ~4 chunks per worker, but keep chunks reasonably sized
        chunk_ = std::max<int64_t>(1, n / (n_threads_ * 4));
        active_ = (int)workers_.size();
        ++generation_;
    }
    cv_start_.notify_all();

    // caller participates
    while (true) {
        int64_t start;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (next_ >= n_) break;
            start = next_;
            next_ = std::min(n_, next_ + chunk_);
        }
        fn(start, std::min(n_, start + chunk_));
    }

    std::unique_lock<std::mutex> lock(mtx_);
    cv_done_.wait(lock, [&] { return active_ == 0; });
    fn_ = nullptr;
}

void ThreadPool::worker_loop() {
    int seen_generation = 0;
    while (true) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_start_.wait(lock, [&] { return stop_ || generation_ != seen_generation; });
        if (stop_) return;
        seen_generation = generation_;
        const auto* fn = fn_;
        const int64_t n = n_;
        const int64_t chunk = chunk_;
        lock.unlock();

        while (true) {
            int64_t start;
            {
                std::lock_guard<std::mutex> lock2(mtx_);
                if (next_ >= n) break;
                start = next_;
                next_ = std::min(n, next_ + chunk);
            }
            (*fn)(start, std::min(n, start + chunk));
        }

        lock.lock();
        if (--active_ == 0) cv_done_.notify_one();
    }
}

} // namespace haidass
