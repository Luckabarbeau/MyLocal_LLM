// Diagnostic only. Build: g++ -O3 -std=c++17 -shared -pthread -fPIC <this file> -o <library>
// No quantization or arithmetic implementation: persistent host+worker row dispatch only.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

using RowCallback = void (*)(const void*, const uint8_t*, size_t, size_t,
                            int, int, int, float*, size_t);
using DotFn = void (*)(int, float*, size_t, const void*, size_t, const void*, size_t, int);
using Q2RowsFn = int (*)(const void*, const uint8_t*, size_t, size_t,
                        int, int, int, float*, size_t);

// Native helper context layouts. Function fields are native C function pointers,
// NEVER Python CFUNCTYPE callbacks. The Q2 error field must initially be zero;
// access it with q2_error below, or only after all callbacks have finished.
struct Strata0335DotContext { DotFn dot; const void* act; };
struct Strata0335Q2Context { Q2RowsFn rows_fn; const void* act; mutable int error; };

#define API extern "C" __attribute__((visibility("default")))
namespace {
// GCC/Clang atomic builtins keep the Q2 context C/ctypes-compatible (a plain int).
void record_error(int* destination, int code) noexcept {
    int expected = 0;
    __atomic_compare_exchange_n(destination, &expected, code, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
}
}

API int strata0335_row_q2_error(const void* context) noexcept {
    if (!context) return 1;
    return __atomic_load_n(&static_cast<const Strata0335Q2Context*>(context)->error, __ATOMIC_RELAXED);
}

API void strata0335_row_dot_callback(const void* context, const uint8_t* weight,
                                     size_t weight_bytes, size_t stride, int k, int r0, int r1,
                                     float* out, size_t out_count) {
    // The team validates bounds. A standalone caller must supply the same contract.
    (void)weight_bytes;
    (void)out_count;
    const auto& c = *static_cast<const Strata0335DotContext*>(context);
    for (int r = r0; r < r1; ++r)
        c.dot(k, out+r, 0, weight+size_t(r)*stride, 0, c.act, 0, 1);
}

API void strata0335_row_q2_callback(const void* context, const uint8_t* weight,
                                    size_t weight_bytes, size_t stride, int k, int r0, int r1,
                                    float* out, size_t out_count) {
    const auto& c = *static_cast<const Strata0335Q2Context*>(context);
    const int code = c.rows_fn(c.act, weight, weight_bytes, stride, k, r0, r1, out, out_count);
    if (code) record_error(&c.error, code);
}

namespace {
// Status values are part of the diagnostic ABI, not GGML or Strata statuses.
enum Status { ok=0, invalid=1, busy=2, callback_failed=3, internal_error=4 };
constexpr int max_experts = 10, max_rows = 65536, max_iterations = 1000000;
struct Job {
    RowCallback callback = nullptr;
    const void* context = nullptr;
    const uint8_t* weights[max_experts]{};
    size_t bytes[max_experts]{};
    int experts = 0, n = 0, k = 0;
    size_t stride = 0, total = 0, chunk = 0;
    float* output = nullptr;
};

class Team {
public:
    explicit Team(int participants) : participants_(participants) {
        workers_.reserve(size_t(participants-1));
        try {
            for (int i=1; i<participants; ++i)
                workers_.emplace_back([this] { worker(); });
        } catch (...) {
            shutdown(); // std::thread destruction must never see a joinable thread.
            throw;
        }
    }
    ~Team() {
        // Wait for an active synchronous run before stopping its workers. The caller
        // must prevent new calls (including queued callers) once destruction begins.
        std::lock_guard<std::mutex> owner(run_mutex_);
        shutdown();
    }
    int last_error() const noexcept { return last_error_.load(std::memory_order_relaxed); }

    int run(RowCallback callback, const void* context, const uint8_t* const* weights,
            const size_t* bytes, int experts, int n, size_t stride, int k,
            float* output, int iterations, double* elapsed) {
        if (elapsed) *elapsed = 0.0;
        std::unique_lock<std::mutex> owner(run_mutex_, std::try_to_lock);
        if (!owner.owns_lock()) return busy;
        last_error_.store(0, std::memory_order_relaxed);
        if (!callback || !weights || !bytes || !output || !elapsed ||
            experts<1 || experts>max_experts || n<1 || n>max_rows || k<1 ||
            !stride || iterations<1 || iterations>max_iterations) return invalid;
        // Without a weight-format argument, require a full stride for every row,
        // including final padding. Callback-specific row geometry remains its contract.
        if (stride > std::numeric_limits<size_t>::max()/size_t(n)) return invalid;
        const size_t required = size_t(n)*stride;
        for (int e=0; e<experts; ++e)
            if (!weights[e] || bytes[e]<required) return invalid;
        if (callback == strata0335_row_dot_callback) {
            if (!context) return invalid;
            const auto& c = *static_cast<const Strata0335DotContext*>(context);
            if (!c.dot || !c.act) return invalid;
        }
        const Strata0335Q2Context* q2 = nullptr;
        if (callback == strata0335_row_q2_callback) {
            if (!context || k%64 || stride<size_t(k/64)*18) return invalid;
            q2 = static_cast<const Strata0335Q2Context*>(context);
            if (!q2->rows_fn || !q2->act) return invalid;
            __atomic_store_n(&q2->error, 0, __ATOMIC_RELAXED);
        }
        Job next_job;
        next_job.callback=callback; next_job.context=context;
        next_job.experts=experts; next_job.n=n; next_job.k=k;
        next_job.stride=stride; next_job.total=size_t(experts)*size_t(n);
        const size_t target=size_t(participants_)*3;
        next_job.chunk=(next_job.total+target-1)/target;
        next_job.output=output;
        for (int e=0; e<experts; ++e) {
            next_job.weights[e]=weights[e]; next_job.bytes[e]=bytes[e];
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_=next_job;
        }
        const auto start=std::chrono::steady_clock::now();
        for (int iteration=0; iteration<iterations; ++iteration) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                next_.store(0, std::memory_order_relaxed);
                remaining_=workers_.size();
                ++epoch_;
            }
            wake_.notify_all();
            consume(); // The calling host is one of the participants.
            {
                std::unique_lock<std::mutex> lock(mutex_);
                done_.wait(lock, [this] { return remaining_==0; });
            }
            if (q2) {
                const int error=__atomic_load_n(&q2->error, __ATOMIC_RELAXED);
                if (error) {
                    int expected=0;
                    last_error_.compare_exchange_strong(expected, error, std::memory_order_relaxed);
                }
            }
            // All participants are drained even when a callback reports a failure.
            if (last_error()) break;
        }
        const auto stop=std::chrono::steady_clock::now();
        *elapsed=std::chrono::duration<double, std::nano>(stop-start).count();
        return last_error() ? callback_failed : ok;
    }

private:
    const int participants_;
    std::vector<std::thread> workers_;
    std::mutex run_mutex_, mutex_;
    std::condition_variable wake_, done_;
    bool stopping_=false;
    uint64_t epoch_=0;
    size_t remaining_=0;
    Job job_;
    std::atomic<size_t> next_{0};
    std::atomic<int> last_error_{0};

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_=true;
        }
        wake_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }
    void consume() noexcept {
        for (;;) {
            const size_t begin=next_.fetch_add(job_.chunk, std::memory_order_relaxed);
            if (begin>=job_.total) return;
            const size_t end=std::min(job_.total, begin+job_.chunk);
            for (size_t position=begin; position<end;) {
                const size_t e=position/size_t(job_.n);
                const int r0=int(position%size_t(job_.n));
                const int count=int(std::min(end-position, size_t(job_.n-r0)));
                try {
                    job_.callback(job_.context, job_.weights[e], job_.bytes[e], job_.stride,
                                  job_.k, r0, r0+count, job_.output+e*size_t(job_.n), size_t(job_.n));
                } catch (...) {
                    // Native C++ callbacks must not strand participants at the barrier.
                    int expected=0;
                    last_error_.compare_exchange_strong(expected, -1, std::memory_order_relaxed);
                }
                position+=size_t(count);
            }
        }
    }
    void worker() {
        uint64_t seen=0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [this, seen] { return stopping_ || epoch_!=seen; });
                if (stopping_) return;
                seen=epoch_;
            }
            consume();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--remaining_==0) done_.notify_one();
            }
        }
    }
};
}

// Native pointers only: ctypes may describe these addresses but MUST NOT supply
// Python-backed callbacks. Buffers/context/function libraries stay alive until run returns.
// Output capacity is caller-owned: allocate at least E*n floats. No output allocation.
API void* strata0335_row_team_create(int participants) noexcept {
    if (participants<1 || participants>24) return nullptr;
    try { return new Team(participants); } catch (...) { return nullptr; }
}
API void strata0335_row_team_destroy(void* handle) noexcept { delete static_cast<Team*>(handle); }
API int strata0335_row_team_last_error(const void* handle) noexcept {
    return handle ? static_cast<const Team*>(handle)->last_error() : 0;
}
API int strata0335_row_team_run(void* handle, RowCallback callback, const void* context,
                                const uint8_t* const* weights, const size_t* bytes,
                                int experts, int n, size_t stride, int k, float* output,
                                int iterations, double* elapsed_ns) noexcept {
    if (elapsed_ns) *elapsed_ns=0.0;
    if (!handle) return invalid;
    try {
        return static_cast<Team*>(handle)->run(callback, context, weights, bytes,
                                               experts, n, stride, k, output, iterations, elapsed_ns);
    } catch (...) { return internal_error; }
}
API int strata0335_row_team_run_dot(void* handle, DotFn dot, const void* act,
                                    const uint8_t* const* weights, const size_t* bytes,
                                    int experts, int n, size_t stride, int k, float* output,
                                    int iterations, double* elapsed_ns) noexcept {
    const Strata0335DotContext context{dot, act};
    return strata0335_row_team_run(handle, strata0335_row_dot_callback, &context,
                                  weights, bytes, experts, n, stride, k, output, iterations, elapsed_ns);
}
API int strata0335_row_team_run_q2(void* handle, Q2RowsFn rows, const void* act,
                                   const uint8_t* const* weights, const size_t* bytes,
                                   int experts, int n, size_t stride, int k, float* output,
                                   int iterations, double* elapsed_ns) noexcept {
    const Strata0335Q2Context context{rows, act, 0};
    return strata0335_row_team_run(handle, strata0335_row_q2_callback, &context,
                                  weights, bytes, experts, n, stride, k, output, iterations, elapsed_ns);
}
