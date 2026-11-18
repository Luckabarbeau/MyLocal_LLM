// Header/vocabulary-only oracle against pinned llama.cpp; no inference weights.
#include "llama.h"
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

static void quiet(enum ggml_log_level, const char *, void *) {}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    llama_log_set(quiet, nullptr);
    llama_backend_init();
    auto params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    auto *model = llama_model_load_from_file(argv[1], params);
    if (!model) return 3;
    const auto *vocab = llama_model_get_vocab(model);
    std::string text((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    std::vector<llama_token> tokens(text.size() + 32);
    int n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), false, true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), false, true);
    }
    if (n < 0) return 4;
    for (int i = 0; i < n; ++i) std::cout << (i ? " " : "") << tokens[i];
    std::cout << '\n';
    llama_model_free(model);
    llama_backend_free();
}
