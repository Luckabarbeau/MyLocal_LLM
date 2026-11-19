// SPDX-License-Identifier: LGPL-3.0-or-later
// Original MyLocal_LLM bridge; no GGML quantization mathematics copied.
// Calls the separately supplied MIT-licensed GGML library through its public
// ggml-cpu.h ABI (reference revision 171e8846b4af9766c354064cb776cb34a50f053f).
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <algorithm>
#include <chrono>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#include "q2_cpu_simd.h"

using Quantize = void (*)(const float *, void *, int64_t);
using Dot = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

struct Projection {
    const uint8_t *weight;
    size_t stride;
    int qtype; // GGML_TYPE_Q8_0=8 or GGML_TYPE_Q8_K=15
    Quantize quantize;
    Dot dot;
};

static bool valid(const Projection &p, int k) {
    return p.weight && p.quantize && p.dot && p.stride && k > 0 &&
        ((p.qtype == 15 && k % 256 == 0) || (p.qtype == 8 && k % 32 == 0));
}

// uint64_t storage gives Q8_K float/int16 fields natural alignment.
static std::vector<uint64_t> scratch(int k, int type) {
    size_t bytes = type == 15 ? size_t(k / 256) * 292 : size_t(k / 32) * 34;
    return std::vector<uint64_t>((bytes + 7) / 8);
}

static void rows(const Projection &p, int n, int k, const void *q, float *out) {
    for (int r = 0; r < n; ++r)
        p.dot(k, out + r, 0, p.weight + size_t(r) * p.stride, 0, q, 0, 1);
}

// Original bounded row-phase scheduler. All workers cross every phase barrier;
// descriptors/scratch cannot be reused while a worker still references them.
struct PoolExpert { Projection gate, up, down; };

class RowPool {
public:
    const int d, f, capacity;
    std::vector<PoolExpert> experts;
    std::vector<PoolExpert> registered;
    std::vector<float> hidden;
    std::vector<std::vector<uint64_t>> qhidden;
    std::vector<uint64_t> input_k, input_0;
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable wake, barrier, done;
    std::atomic<size_t> next{0};
    alignas(64) std::atomic<size_t> epoch{0};
    alignas(64) std::atomic<int> phase{0};
    std::atomic<int> spin_us{0};
    std::atomic<bool> host_help{false}, task_mode{false}, fused_hidden{false};
    alignas(64) std::atomic<uint64_t> task_queue{0};
    alignas(64) std::atomic<uint32_t> task_remaining{0}, task_total{0}, task_chunk{1};
    static constexpr uint64_t task_mask = (uint64_t(1)<<22)-1;
    int arrived = 0, count = 0, assisting = 0;
    std::atomic<bool> busy{false}, stopping{false};
    float *output = nullptr;

    RowPool(int dim, int width, int cap, int threads)
        : d(dim), f(width), capacity(cap), experts(cap), hidden(size_t(cap)*width),
          qhidden(cap), input_k(scratch(dim, 15)), input_0(scratch(dim, 8)) {
        for (auto &q : qhidden) q = scratch(width, 15);
        // Q8_0 can require more bytes than Q8_K for short, non-256 widths.
        const size_t words = std::max(scratch(width, 15).size(), scratch(width, 8).size());
        for (auto &q : qhidden) q.resize(words);
        registered.reserve(32768); // Metadata only, never packed weight bytes.
        workers.reserve(threads);
        try {
            for (int i = 0; i < threads; ++i) workers.emplace_back([this]{ run(); });
        } catch (...) {
            { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
            wake.notify_all();
            for (auto &worker : workers) worker.join();
            throw;
        }
    }
    ~RowPool() {
        wait();
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        wake.notify_all();
        for (auto &worker : workers) worker.join();
    }
    const void *input(const Projection &p) const {
        return p.qtype == 15 ? input_k.data() : input_0.data();
    }
    int submit(const PoolExpert *source, int n, const float *x, float *out) {
        std::lock_guard<std::mutex> lock(mutex);
        return begin(source,n,x,out);
    }
    int register_expert(const PoolExpert *source, int *index) {
        std::lock_guard<std::mutex> lock(mutex);
        if (busy) return 4;
        if (!source || !index || registered.size() >= 32768 ||
                !valid(source->gate,d) || !valid(source->up,d) || !valid(source->down,f)) return 1;
        *index = int(registered.size());
        registered.push_back(*source);
        return 0;
    }
    int submit_registered(const int *indices, int n, const float *x, float *out) {
        std::lock_guard<std::mutex> lock(mutex);
        if (busy) return 4;
        if (!indices || n <= 0 || n > capacity) return 1;
        for (int i=0; i<n; ++i)
            if (indices[i]<0 || size_t(indices[i])>=registered.size()) return 1;
        for (int i=0; i<n; ++i) experts[i] = registered[indices[i]];
        return begin(experts.data(),n,x,out);
    }
    int begin(const PoolExpert *source, int n, const float *x, float *out) {
        if (busy) return 4;
        if (!source || !x || !out || n <= 0 || n > capacity) return 1;
        Quantize quant_k = nullptr, quant_0 = nullptr;
        for (int i = 0; i < n; ++i) {
            const auto &e = source[i];
            if (!valid(e.gate, d) || !valid(e.up, d) || !valid(e.down, f)) return 1;
            for (const auto &p : {e.gate, e.up}) {
                auto &quant = p.qtype == 15 ? quant_k : quant_0;
                if (quant && quant != p.quantize) return 1;
                quant = p.quantize;
            }
        }
        if (quant_k) quant_k(x, input_k.data(), d);
        if (quant_0) quant_0(x, input_0.data(), d);
        if (task_mode && epoch.load() >= (uint64_t(1)<<40)-1) return 6;
        if (source != experts.data()) std::copy(source, source+n, experts.begin());
        count = n; output = out; phase = 0; arrived = 0; next.store(0);
        busy = true; ++epoch;
        if (task_mode) publish_task_phase(0);
        wake.notify_all();
        return 0;
    }
    template<class Predicate> void bounded_spin(Predicate ready) {
        const int microseconds = spin_us.load(std::memory_order_relaxed);
        if (!microseconds || ready()) return;
        const auto deadline = std::chrono::steady_clock::now()+std::chrono::microseconds(microseconds);
        do {
            for (int i = 0; i < 128; ++i) {
                if (ready()) return;
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#else
                std::this_thread::yield();
#endif
            }
        } while (std::chrono::steady_clock::now() < deadline);
    }
    void advance_phase(int p) {
        // Called with mutex held. A host task is a live phase consumer too.
        if (arrived != int(workers.size()) || assisting) return;
        arrived = 0; next.store(0); ++phase;
        if (p == 2) { busy = false; done.notify_all(); }
        barrier.notify_all();
    }
    void publish_task_phase(int p) {
        const uint64_t tag = (uint64_t(epoch.load())*4+p)<<22;
        // An exhausted sentinel closes the previous queue before counters change.
        task_queue.store(tag|task_mask, std::memory_order_release);
        const size_t rows = p==1 ? size_t(count) : size_t(count)*(p==0 ? f : d);
        const size_t target = 3*(workers.size()+(host_help.load()?1:0));
        const size_t chunk = p==1 ? 1 : std::max<size_t>(16,(((rows+target-1)/target+15)/16)*16);
        task_chunk.store(uint32_t(chunk));
        const uint32_t tasks = uint32_t((rows+chunk-1)/chunk);
        task_total.store(tasks); task_remaining.store(tasks); phase.store(p);
        task_queue.store(tag, std::memory_order_release);
        barrier.notify_all();
    }
    void task_worker(size_t batch_epoch) {
        while (busy.load(std::memory_order_acquire)) {
            uint64_t claim = task_queue.load(std::memory_order_acquire);
            const uint64_t tag = claim>>22;
            if (tag/4 != batch_epoch) return;
            const uint32_t index = uint32_t(claim&task_mask);
            if (index >= task_total.load(std::memory_order_acquire)) {
                bounded_spin([this,claim]{ return !busy.load() || task_queue.load()!=claim; });
                std::unique_lock<std::mutex> lock(mutex);
                barrier.wait(lock,[this,claim]{ return !busy || task_queue.load()!=claim; });
                continue;
            }
            if (!task_queue.compare_exchange_weak(claim,claim+1,std::memory_order_acq_rel)) continue;
            // Only a successful generation-tagged claim may read descriptors.
            // This task remains counted until all of its array writes retire.
            const int p = int(tag%4);
            const size_t chunk = task_chunk.load();
            const size_t rows = p==1 ? size_t(count) : size_t(count)*(p==0 ? f : d);
            compute_range(p,size_t(index)*chunk,std::min(size_t(index+1)*chunk,rows));
            if (task_remaining.fetch_sub(1,std::memory_order_acq_rel)==1) {
                std::lock_guard<std::mutex> lock(mutex);
                task_queue.store((tag<<22)|task_mask,std::memory_order_release);
                if (p==0 && fused_hidden.load(std::memory_order_relaxed)) {
                    // The final gate task acquires every gate write through the
                    // completion counter. Quantize the bounded hidden rows here,
                    // avoiding a mostly idle worker phase for <=capacity rows.
                    compute_range(1,0,size_t(count));
                    publish_task_phase(2);
                } else if (p<2) publish_task_phase(p+1);
                else { phase=3; busy=false; done.notify_all(); barrier.notify_all(); }
            }
        }
    }
    void wait() {
        if (task_mode && host_help) { task_worker(epoch.load()); return; }
        if (!host_help.load(std::memory_order_relaxed)) {
            bounded_spin([this]{ return !busy.load(std::memory_order_acquire); });
            std::unique_lock<std::mutex> lock(mutex);
            done.wait(lock, [this]{ return !busy; });
            return;
        }
        std::unique_lock<std::mutex> lock(mutex);
        while (busy) {
            const int p = phase.load();
            const size_t total = p == 1 ? size_t(count) : size_t(count)*(p == 0 ? f : d);
            if (next.load() < total) {
                ++assisting;
                lock.unlock(); phase_work(p); lock.lock();
                --assisting;
                advance_phase(p);
            } else {
                lock.unlock();
                bounded_spin([this, p]{ return !busy.load() || phase.load() != p; });
                lock.lock();
                barrier.wait(lock, [this, p]{ return !busy || phase != p; });
            }
        }
    }
    void phase_work(int p) {
        const size_t total = p == 1 ? size_t(count) : size_t(count)*(p == 0 ? f : d);
        const size_t chunk = p == 1 ? 1 : 16;
        for (;;) {
            const size_t start = next.fetch_add(chunk, std::memory_order_relaxed);
            if (start >= total) break;
            compute_range(p,start,std::min(start+chunk,total));
        }
    }
    void compute_range(int p, size_t start, size_t end) {
            for (size_t task = start; task < end; ++task) {
                const int e = p == 1 ? int(task) : int(task / (p == 0 ? f : d));
                const int row = p == 1 ? 0 : int(task % (p == 0 ? f : d));
                const auto &w = experts[e];
                if (p == 0) {
                    float g, u;
                    w.gate.dot(d, &g, 0, w.gate.weight+size_t(row)*w.gate.stride, 0,
                               input(w.gate), 0, 1);
                    w.up.dot(d, &u, 0, w.up.weight+size_t(row)*w.up.stride, 0,
                             input(w.up), 0, 1);
                    hidden[size_t(e)*f+row] = (g/(1.0f+std::exp(-g)))*u;
                } else if (p == 1) {
                    w.down.quantize(hidden.data()+size_t(e)*f, qhidden[e].data(), f);
                } else {
                    w.down.dot(f, output+size_t(e)*d+row, 0,
                               w.down.weight+size_t(row)*w.down.stride, 0,
                               qhidden[e].data(), 0, 1);
                }
            }
    }
    void run() {
        size_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex);
            if (spin_us.load(std::memory_order_relaxed)) {
                lock.unlock();
                bounded_spin([this, seen]{ return stopping.load(std::memory_order_acquire) ||
                    epoch.load(std::memory_order_acquire) != seen; });
                lock.lock();
            }
            wake.wait(lock, [this, seen]{ return stopping || epoch != seen; });
            if (stopping) return;
            seen = epoch;
            if (task_mode) { lock.unlock(); task_worker(seen); continue; }
            for (int p = 0; p < 3; ++p) {
                lock.unlock(); phase_work(p); lock.lock();
                ++arrived;
                if (arrived == int(workers.size()) && !assisting) {
                    advance_phase(p);
                } else {
                    if (spin_us.load(std::memory_order_relaxed)) {
                        lock.unlock();
                        bounded_spin([this, p]{ return phase.load(std::memory_order_acquire) != p; });
                        lock.lock();
                    }
                    barrier.wait(lock, [this, p]{ return phase != p; });
                }
            }
        }
    }
};

extern "C" void *mylocal_ggml_pool_create(int d, int f, int capacity, int threads) noexcept {
    if (d <= 0 || f <= 0 || d % 32 || f % 32 || d > 65536 || f > 65536 ||
        capacity <= 0 || capacity > 512 || threads <= 0 || threads > 256) return nullptr;
    try { return new RowPool(d, f, capacity, threads); } catch (...) { return nullptr; }
}
extern "C" int mylocal_ggml_pool_set_spin(void *pool, int microseconds) noexcept {
    if (!pool || microseconds < 0 || microseconds > 2000) return 1;
    auto &p = *static_cast<RowPool *>(pool);
    std::lock_guard<std::mutex> lock(p.mutex);
    if (p.busy) return 4;
    p.spin_us.store(microseconds, std::memory_order_relaxed);
    return 0;
}
extern "C" int mylocal_ggml_pool_set_task_mode(void *pool, int enabled) noexcept {
    if (!pool || (enabled!=0 && enabled!=1)) return 1;
    auto &p = *static_cast<RowPool *>(pool);
    std::lock_guard<std::mutex> lock(p.mutex);
    if (p.busy) return 4;
    p.task_mode.store(enabled);
    return 0;
}
extern "C" int mylocal_ggml_pool_set_fused_hidden(void *pool, int enabled) noexcept {
    if (!pool || (enabled!=0 && enabled!=1)) return 1;
    auto &p = *static_cast<RowPool *>(pool);
    std::lock_guard<std::mutex> lock(p.mutex);
    if (p.busy) return 4;
    p.fused_hidden.store(enabled,std::memory_order_relaxed);
    return 0;
}
extern "C" int mylocal_ggml_pool_configure(void *pool, int help, const int *cpus, int n) noexcept {
    if (!pool || (help != 0 && help != 1)) return 1;
    auto &p = *static_cast<RowPool *>(pool);
    std::lock_guard<std::mutex> lock(p.mutex);
    if (p.busy) return 4;
    if (n && (n != int(p.workers.size()) || !cpus)) return 1;
#if defined(__linux__)
    for (int i = 0; i < n; ++i) if (cpus[i] < 0 || cpus[i] >= CPU_SETSIZE) return 1;
    for (int i = 0; i < n; ++i) {
        cpu_set_t mask;
        CPU_ZERO(&mask); CPU_SET(cpus[i], &mask);
        if (pthread_setaffinity_np(p.workers[i].native_handle(), sizeof(mask), &mask)) return 5;
    }
#else
    if (n) return 5;
#endif
    p.host_help.store(help, std::memory_order_relaxed);
    return 0;
}
extern "C" int mylocal_ggml_pool_submit(void *pool, const PoolExpert *experts,
                                       int count, const float *x, float *out) noexcept {
    if (!pool) return 1;
    try { return static_cast<RowPool *>(pool)->submit(experts, count, x, out); }
    catch (...) { return 3; }
}
extern "C" int mylocal_ggml_pool_register(void *pool, const PoolExpert *expert, int *index) noexcept {
    if (!pool) return 1;
    try { return static_cast<RowPool *>(pool)->register_expert(expert,index); } catch (...) { return 3; }
}
extern "C" int mylocal_ggml_pool_submit_registered(void *pool, const int *indices, int n,
                                                    const float *x, float *out) noexcept {
    if (!pool) return 1;
    try { return static_cast<RowPool *>(pool)->submit_registered(indices,n,x,out); } catch (...) { return 3; }
}
extern "C" void mylocal_ggml_pool_wait(void *pool) noexcept {
    if (pool) static_cast<RowPool *>(pool)->wait();
}
extern "C" void mylocal_ggml_pool_destroy(void *pool) noexcept {
    delete static_cast<RowPool *>(pool);
}

// A CUDA host node calls this original bridge without entering Python or CUDA.
// D2H/event dependencies publish the inputs; H2D/event dependencies consume the
// output. One serialized model stream owns each handoff and the shared row pool.
struct NativeHandoff {
    RowPool *pool;
    const float *input;
    const int *metadata;
    float *output;
    int k;
    std::vector<int> registry;
    uint64_t runs = 0, misses = 0, hits = 0, errors = 0, elapsed_ns = 0, batches = 0;
    NativeHandoff(RowPool *p, const float *x, const int *m, float *y, int n,
                  const int *ids, int experts)
        : pool(p), input(x), metadata(m), output(y), k(n), registry(ids,ids+experts) {}
};
extern "C" void *mylocal_ggml_handoff_create(void *pool, const float *x, const int *meta,
        float *out, int k, const int *registry, int n) noexcept {
    if (!pool || !x || !meta || !out || !registry || k<1 || k>16 || n<k || n>512 ||
            k>static_cast<RowPool *>(pool)->capacity) return nullptr;
    try { return new NativeHandoff(static_cast<RowPool *>(pool),x,meta,out,k,registry,n); }
    catch (...) { return nullptr; }
}
extern "C" void mylocal_ggml_handoff_run(void *handle) noexcept {
    auto &h = *static_cast<NativeHandoff *>(handle);
    int indices[16], count=0, error=0, previous=-1;
    const auto start=std::chrono::steady_clock::now();
    ++h.runs;
    try {
        if (h.metadata[h.k]) error=1;
        for (int i=0; i<h.k && !error; ++i) {
            const int id=h.metadata[i];
            if (id<=previous || id<0 || size_t(id)>=h.registry.size()) { error=1; break; }
            previous=id;
            if (h.registry[id]>=0) indices[count++]=h.registry[id];
        }
        if (!error && count) {
            error=h.pool->submit_registered(indices,count,h.input,h.output);
            if (!error) h.pool->wait();
        }
    } catch (...) { error=3; }
    if (error) {
        ++h.errors;
        // Poison, never silently reuse stale CPU rows on an invalid epoch.
        std::fill(h.output,h.output+size_t(h.k)*h.pool->d,NAN);
    } else { h.misses+=count; h.hits+=h.k-count; h.batches+=bool(count); }
    h.elapsed_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-start).count();
}
extern "C" void mylocal_ggml_handoff_stats(void *handle, uint64_t *out) noexcept {
    if (!handle || !out) return;
    const auto &h=*static_cast<NativeHandoff *>(handle);
    out[0]=h.runs; out[1]=h.misses; out[2]=h.hits; out[3]=h.errors;
    out[4]=h.elapsed_ns; out[5]=h.batches;
}
extern "C" void mylocal_ggml_handoff_destroy(void *handle) noexcept {
    delete static_cast<NativeHandoff *>(handle);
}

extern "C" int mylocal_ggml_linear(
    const uint8_t *w, size_t stride, int type, Quantize quantize, Dot dot,
    int n, int k, const float *x, int m, float *out) noexcept {
    Projection p{w, stride, type, quantize, dot};
    if (!valid(p, k) || n <= 0 || m < 0 || !x || !out) return 1;
    try {
        auto q = scratch(k, type);
        for (int b = 0; b < m; ++b) {
            quantize(x + size_t(b) * k, q.data(), k);
            rows(p, n, k, q.data(), out + size_t(b) * n);
        }
        return 0;
    } catch (const std::bad_alloc &) { return 2; }
      catch (...) { return 3; }
}

extern "C" int mylocal_ggml_expert(
    const uint8_t *gw, size_t gs, int gt, Quantize gq, Dot gd,
    const uint8_t *uw, size_t us, int ut, Quantize uq, Dot ud,
    const uint8_t *dw, size_t ds, int dt, Quantize dq, Dot dd,
    int d, int f, const float *x, int m, float *out) noexcept {
    Projection gate{gw, gs, gt, gq, gd}, up{uw, us, ut, uq, ud}, down{dw, ds, dt, dq, dd};
    if (!valid(gate, d) || !valid(up, d) || !valid(down, f) ||
        f <= 0 || d <= 0 || m < 0 || !x || !out) return 1;
    try {
        auto qg = scratch(d, gt);
        auto qu = scratch(gt == ut ? 0 : d, ut);
        auto qh = scratch(f, dt);
        std::vector<float> g(f), u(f), h(f);
        for (int b = 0; b < m; ++b) {
            const float *input = x + size_t(b) * d;
            gq(input, qg.data(), d);
            const void *up_input = qg.data();
            if (gt != ut) { uq(input, qu.data(), d); up_input = qu.data(); }
            rows(gate, f, d, qg.data(), g.data());
            rows(up, f, d, up_input, u.data());
            for (int j = 0; j < f; ++j)
                h[j] = (g[j] / (1.0f + std::exp(-g[j]))) * u[j];
            dq(h.data(), qh.data(), f);
            rows(down, d, f, qh.data(), out + size_t(b) * d);
        }
        return 0;
    } catch (const std::bad_alloc &) { return 2; }
      catch (...) { return 3; }
}
