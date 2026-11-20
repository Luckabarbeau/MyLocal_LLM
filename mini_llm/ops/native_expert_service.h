// SPDX-License-Identifier: LGPL-3.0-or-later
// Original mapped-doorbell service. Included after NativeHandoff's definition.
// The service enters neither CUDA nor Python, including optional packed staging.
// Arithmetic stays in the existing pool; GPU kernels pull published host bytes.
struct alignas(64) ExpertDoorbell {
    alignas(64) std::atomic<uint32_t> request{0};
    alignas(64) std::atomic<uint32_t> done{0};
};
static_assert(offsetof(ExpertDoorbell,done)==64 && sizeof(ExpertDoorbell)==128,"doorbell ABI");

class NativeExpertService {
    struct Entry { NativeHandoff *handoff; ExpertDoorbell *flags; NativeExpertStage *stage=nullptr; };
    RowPool *pool;
    std::atomic<uint32_t> *failure;
    std::mutex mutex;
    std::vector<Entry> entries;
    std::atomic<bool> stopping{false};
    std::thread thread;
public:
    NativeExpertService(RowPool *p,uint32_t *error) : pool(p),
        failure(new(error) std::atomic<uint32_t>(0)) {
        entries.reserve(64);
        thread=std::thread([this]{run();});
    }
    ~NativeExpertService() {
        stopping.store(true,std::memory_order_release);
        thread.join(); // Also retires an already submitted CPU batch.
    }
    int add(NativeHandoff *h,void *flags) {
        if (!h || h->pool!=pool || !flags || uintptr_t(flags)%64) return 1;
        std::lock_guard<std::mutex> lock(mutex);
        if (entries.size()>=64) return 1;
        for (const auto &e:entries) if(e.handoff==h || e.flags==flags) return 1;
        auto *f=new(flags) ExpertDoorbell;
        entries.push_back({h,f});
        return 0;
    }
    int set_stage(NativeHandoff *h,NativeExpertStage *stage) {
        if(!stage || stage->h!=h) return 1;
        std::lock_guard<std::mutex> lock(mutex);
        for(auto &e:entries) if(e.handoff==h && !e.stage) { e.stage=stage; return 0; }
        return 1;
    }
    int remove(NativeHandoff *h) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it=entries.begin();it!=entries.end();++it) {
            if(it->handoff==h) { entries.erase(it); return 0; }
        }
        return 1;
    }
    void run() noexcept {
        while (!stopping.load(std::memory_order_acquire)) {
            {
                // Registration/removal are setup/retirement seams. Holding this
                // through a batch prevents a handoff or its slabs being freed.
                std::lock_guard<std::mutex> lock(mutex);
                if (!failure->load(std::memory_order_acquire)) {
                    for (auto &e:entries) {
                        const uint32_t request=e.flags->request.load(std::memory_order_acquire);
                        if (request==e.flags->done.load(std::memory_order_relaxed)) continue;
                        const auto errors=e.handoff->errors;
                        if(e.stage) e.stage->run(request,failure);
                        else mylocal_ggml_handoff_run(e.handoff);
                        if(e.handoff->errors!=errors) failure->store(2,std::memory_order_release);
                        e.flags->done.store(request,std::memory_order_release);
                        break; // The model's serial residual chain publishes one ready batch.
                    }
                }
            }
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#else
            std::this_thread::yield();
#endif
        }
    }
};
extern "C" void *mylocal_ggml_service_create(void *pool,uint32_t *failure) noexcept {
    if(!pool || !failure || uintptr_t(failure)%64) return nullptr;
    try { return new NativeExpertService(static_cast<RowPool*>(pool),failure); }
    catch(...) { return nullptr; }
}
extern "C" int mylocal_ggml_service_add(void *service,void *handoff,void *flags) noexcept {
    if(!service) return 1;
    try { return static_cast<NativeExpertService*>(service)->add(static_cast<NativeHandoff*>(handoff),flags); }
    catch(...) { return 3; }
}
extern "C" int mylocal_ggml_service_remove(void *service,void *handoff) noexcept {
    if(!service) return 1;
    return static_cast<NativeExpertService*>(service)->remove(static_cast<NativeHandoff*>(handoff));
}
extern "C" int mylocal_ggml_service_set_stage(void *service,void *handoff,void *stage) noexcept {
    if(!service || !handoff) return 1;
    try { return static_cast<NativeExpertService*>(service)->set_stage(
        static_cast<NativeHandoff*>(handoff),static_cast<NativeExpertStage*>(stage)); }
    catch(...) { return 3; }
}
extern "C" void mylocal_ggml_service_destroy(void *service) noexcept {
    delete static_cast<NativeExpertService*>(service);
}
