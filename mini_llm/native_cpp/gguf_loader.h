#pragma once
#include "runtime.h"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace ml {
struct GGUFError : std::runtime_error { using std::runtime_error::runtime_error; };
struct Mapping {
    const unsigned char* data=nullptr;
    uint64_t size=0;
    explicit Mapping(const char* path);
    ~Mapping();
    Mapping(const Mapping&)=delete;
};
struct Metadata { std::string key; uint32_t type; std::vector<unsigned char> value; uint64_t shard; };
struct Tensor { std::string name; ml_gguf_tensor desc{}; };
struct GGUF {
    std::vector<std::unique_ptr<Mapping>> maps;
    std::vector<Metadata> metadata;
    std::vector<Tensor> tensors;
    GGUF(const char* const* paths, uint64_t count);
};
}
