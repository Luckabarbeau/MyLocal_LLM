// Independent logits export through the pinned llama.cpp C API, not production inference.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <link.h>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static std::string json_string(const std::string & value) {
    std::string out = "\"";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { out += '\\'; out += char(c); }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 32) throw std::runtime_error("Unsupported control character in metadata");
        else out += char(c);
    }
    return out + "\"";
}

static int collect_libraries(dl_phdr_info * info, size_t, void * context) {
    const std::string path(info->dlpi_name);
    const std::string name = std::filesystem::path(path).filename().string();
    if (name.rfind("libllama.so", 0) == 0 || name.rfind("libggml", 0) == 0)
        static_cast<std::vector<std::string> *>(context)->push_back(path);
    return 0;
}

struct Backend {
    Backend() { llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
};

struct Resources {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    llama_batch batch{};
    bool has_batch = false;
    ~Resources() {
        if (has_batch) llama_batch_free(batch);
        if (context) llama_free(context);
        if (model) llama_model_free(model);
    }
};

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "--model GGUF --tokens FILE --output BIN --chunk N --threads N --context N\n";
            return 0;
        }
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::runtime_error("Every option requires a value");
            std::string key(argv[i]);
            if (key != "--model" && key != "--tokens" && key != "--output" &&
                key != "--chunk" && key != "--threads" && key != "--context")
                throw std::runtime_error("Unknown option " + key);
            if (!args.emplace(key, argv[i + 1]).second) throw std::runtime_error("Duplicate option " + key);
        }
        for (const char * key : {"--model", "--tokens", "--output", "--chunk", "--threads", "--context"})
            if (!args.count(key)) throw std::runtime_error(std::string("Missing ") + key);
        const int chunk = std::stoi(args.at("--chunk"));
        const int threads = std::stoi(args.at("--threads"));
        const int context_size = std::stoi(args.at("--context"));
        if (chunk < 1 || chunk > 64 || threads < 1 || threads > 8 || context_size < 1 || context_size > 128)
            throw std::runtime_error("Require chunk 1..64, threads 1..8, context 1..128");
        if (std::filesystem::exists(args.at("--output")))
            throw std::runtime_error("Refusing to overwrite existing reference output");
        std::ifstream input(args.at("--tokens"));
        if (!input) throw std::runtime_error("Cannot open token file");
        std::vector<llama_token> tokens;
        long long value;
        while (input >> value) {
            if (value < 0 || value > std::numeric_limits<int32_t>::max() || tokens.size() >= 64)
                throw std::runtime_error("Invalid token ID or more than 64 tokens");
            tokens.push_back(static_cast<llama_token>(value));
        }
        if (!input.eof() || tokens.empty() || tokens.size() > static_cast<size_t>(context_size))
            throw std::runtime_error("Malformed/empty token file or context exceeded");
        const uint32_t endian = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian) != 1)
            throw std::runtime_error("Binary protocol requires little endian");

        Backend backend;
        Resources owned;
        ggml_backend_dev_t devices[] = {nullptr};
        llama_model_params mp = llama_model_default_params();
        mp.devices = devices;
        mp.n_gpu_layers = 0;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        mp.lazy_mode = LLAMA_LAZY_MODE_ON;
        mp.use_extra_bufts = false;
        mp.load_mtp = false;
        owned.model = llama_model_load_from_file(args.at("--model").c_str(), mp);
        if (!owned.model) throw std::runtime_error("llama.cpp model load failed");
        const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(owned.model));
        if (vocab < 1 || vocab > 2000000) throw std::runtime_error("Invalid/unbounded vocabulary");
        for (llama_token token : tokens)
            if (token >= vocab) throw std::runtime_error("Token ID outside reference vocabulary");
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = context_size;
        cp.n_batch = static_cast<uint32_t>(tokens.size());
        cp.n_ubatch = static_cast<uint32_t>(tokens.size());
        cp.n_seq_max = 1;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.n_outputs_max = 0;
        cp.n_outputs_max_per_seq = 0;
        cp.offload_kqv = false;
        cp.op_offload = false;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
        cp.embeddings = false;
        cp.samplers = nullptr;
        cp.n_samplers = 0;
        owned.context = llama_init_from_model(owned.model, cp);
        if (!owned.context) throw std::runtime_error("llama.cpp context creation failed");
        owned.batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
        owned.has_batch = true;
        std::vector<float> logits(tokens.size() * static_cast<size_t>(vocab));
        const auto start = std::chrono::steady_clock::now();
        for (size_t offset = 0; offset < tokens.size();) {
            const int count = std::min(static_cast<size_t>(chunk), tokens.size() - offset);
            owned.batch.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                owned.batch.token[i] = tokens[offset + i];
                owned.batch.pos[i] = static_cast<llama_pos>(offset + i);
                owned.batch.n_seq_id[i] = 1;
                owned.batch.seq_id[i][0] = 0;
                owned.batch.logits[i] = 1;
            }
            const int rc = llama_decode(owned.context, owned.batch);
            if (rc != 0) throw std::runtime_error("llama_decode failed, rc=" + std::to_string(rc));
            for (int i = 0; i < count; ++i) {
                const float * row = llama_get_logits_ith(owned.context, i);
                if (!row) throw std::runtime_error("Missing flagged logits row");
                for (int j = 0; j < vocab; ++j) {
                    if (!std::isfinite(row[j])) throw std::runtime_error("Nonfinite reference logit");
                    logits[(offset + i) * static_cast<size_t>(vocab) + j] = row[j];
                }
            }
            offset += count;
        }
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::ofstream output(args.at("--output"), std::ios::binary);
        const uint32_t rows = static_cast<uint32_t>(tokens.size()), columns = static_cast<uint32_t>(vocab);
        output.write("MLLQREF1", 8);
        output.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        output.write(reinterpret_cast<const char *>(&columns), sizeof(columns));
        output.write(reinterpret_cast<const char *>(tokens.data()), tokens.size() * sizeof(int32_t));
        output.write(reinterpret_cast<const char *>(logits.data()), logits.size() * sizeof(float));
        if (!output) throw std::runtime_error("Failed to write reference output");
        output.close();
        std::vector<std::string> libraries;
        dl_iterate_phdr(collect_libraries, &libraries);
        std::cout << "{\"rows\":" << rows << ",\"vocab\":" << vocab
                  << ",\"chunk\":" << chunk << ",\"threads\":" << threads
                  << ",\"context\":" << context_size << ",\"decode_wall_ms\":" << milliseconds
                  << ",\"cpu_only_execution\":true,\"kv_dtype\":\"f32\",\"flash_attention\":false"
                  << ",\"system_info\":" << json_string(llama_print_system_info())
                  << ",\"loaded_reference_libraries\":[";
        for (size_t i = 0; i < libraries.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << json_string(libraries[i]);
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "0106Q reference error: " << error.what() << '\n';
        return 1;
    }
}
