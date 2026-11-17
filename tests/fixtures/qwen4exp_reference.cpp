// Independent full-model driver for the locally pinned llama.cpp API.
// Original harness code; no GGML kernels or model mathematics are copied here.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>

struct Capture {
    std::string directory;
    int position = -1;
    std::map<std::string, int> occurrences;
    std::set<std::string> asked_names;
    bool failed = false;
};

static bool selected(const std::string & name) {
    return name.rfind("hc_mixed-", 0) == 0 ||
           name.rfind("hc_combine-", 0) == 0 ||
           name.rfind("l_last-", 0) == 0 || name == "result_norm";
}

static void save(const std::string & path, const float * data, size_t count) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data), count * sizeof(float));
    if (!out) throw std::runtime_error("cannot write " + path);
}

static bool callback(ggml_tensor * tensor, bool ask, void * opaque) {
    auto & c = *static_cast<Capture *>(opaque);
    const std::string name = ggml_get_name(tensor);
    if (ask) {
        c.asked_names.insert(name); // No tensor reads during the scheduling query.
        return selected(name);
    }
    if (c.position < 0 || !selected(name)) return true;
    try {
        if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) ||
                ggml_nelements(tensor) > 65536) {
            throw std::runtime_error("unsupported diagnostic layout: " + name);
        }
        std::string stage;
        int layer = -1;
        if (name == "result_norm") stage = "hc_output_read";
        else {
            layer = std::stoi(name.substr(name.find_last_of('-') + 1));
            if (name.rfind("hc_mixed-", 0) == 0) {
                const int occurrence = c.occurrences[name]++;
                if (occurrence > 1) throw std::runtime_error("ambiguous HC boundary");
                stage = occurrence == 0 ? "hc_attention_read" : "hc_ffn_read";
            } else if (name.rfind("l_last-", 0) == 0) stage = "post_moe";
            else stage = "post_mixer";
        }
        std::vector<float> values(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
        save(c.directory + "/p" + std::to_string(c.position) + "_l" +
             std::to_string(layer) + "_" + stage + ".f32", values.data(), values.size());
    } catch (const std::exception & e) {
        std::cerr << "diagnostic capture failed: " << e.what() << '\n';
        c.failed = true;
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    // This revision has no public no-prefetch switch. NUMA is the loader's
    // supported path that disables prefetch (including MAP_POPULATE).
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    const bool demand_paging = ggml_is_numa();
    std::cout << "{\"q2_0_type\":" << int(GGML_TYPE_Q2_0)
              << ",\"numa_disables_prefetch\":" << (demand_paging ? "true" : "false")
              << "}" << std::endl;
    if (argc == 2 && std::string(argv[1]) == "--capabilities") {
        llama_backend_free();
        return 0;
    }
    if (argc != 5) {
        std::cerr << "usage: harness MODEL OUTPUT_DIRECTORY THREADS GREEDY_COUNT(0..3)\n";
        return 2;
    }
    if (!demand_paging) {
        std::cerr << "BLOCKED: pinned loader prefetches non-lazy packed expert tensors on this non-NUMA host\n";
        return 77;
    }
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    llama_batch batch = {};
    try {
        const int threads = std::stoi(argv[3]), greedy_count = std::stoi(argv[4]);
        if (threads < 1 || threads > 32 || greedy_count < 0 || greedy_count > 3)
            throw std::runtime_error("invalid bounded execution arguments");
        Capture capture;
        capture.directory = argv[2];
        auto mp = llama_model_default_params();
        ggml_backend_dev_t devices[] = {nullptr};
        mp.devices = devices;
        mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP; // mmap true, mlock false in this API
        mp.lazy_mode = LLAMA_LAZY_MODE_ON; // PLE gather table remains demand-paged
        mp.use_extra_bufts = false; // No CPU repacking/materialized weight banks
        mp.check_tensors = false; // No full checkpoint scan
        model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("llama_model_load_from_file failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 32;
        cp.n_seq_max = 1;
        cp.n_batch = cp.n_ubatch = 1;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = cp.op_offload = false;
        cp.cb_eval = callback;
        cp.cb_eval_user_data = &capture;
        context = llama_init_from_model(model, cp);
        if (!context) throw std::runtime_error("llama_init_from_model failed");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        batch = llama_batch_init(1, 0, 1);
        std::vector<int> greedy;
        int token = 1;
        // Both fixed input positions request logits; batching is intentionally
        // sequential, matching the production recurrent state update contract.
        for (int p = 0; p < 2 + std::max(0, greedy_count - 1); ++p) {
            if (p == 1) token = 2;
            capture.position = p;
            capture.occurrences.clear();
            batch.n_tokens = 1;
            batch.token[0] = token;
            batch.pos[0] = p;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            if (llama_decode(context, batch) != 0 || capture.failed)
                throw std::runtime_error("llama_decode/capture failed at position " + std::to_string(p));
            llama_synchronize(context);
            const float * logits = llama_get_logits_ith(context, 0);
            if (!logits) throw std::runtime_error("missing requested full logits");
            save(capture.directory + "/logits_" + std::to_string(p) + ".f32", logits, vocab);
            token = std::max_element(logits, logits + vocab) - logits;
            if (p >= 1 && greedy_count) greedy.push_back(token);
        }
        std::ofstream names(capture.directory + "/callback_names.txt");
        for (const auto & name : capture.asked_names) names << name << '\n';
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        std::ofstream stats(capture.directory + "/stats.json");
        stats << "{\"vocab\":" << vocab << ",\"peak_rss_bytes\":"
              << usage.ru_maxrss * 1024LL << ",\"greedy_ids\":[";
        for (size_t i = 0; i < greedy.size(); ++i) stats << (i ? "," : "") << greedy[i];
        stats << "]}";
        llama_batch_free(batch);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        if (batch.token) llama_batch_free(batch);
        if (context) llama_free(context);
        if (model) llama_model_free(model);
        llama_backend_free();
        return 1;
    }
}
