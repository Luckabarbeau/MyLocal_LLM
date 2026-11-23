#include <stdexcept>
#include "gguf_loader.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <bit>
#include <cstring>
#include <algorithm>
#include <map>
#include <set>
#include <limits>
namespace ml {
namespace {
[[noreturn]] void bad(const char* s) { throw GGUFError(s); }
uint64_t add(uint64_t a,uint64_t b) { if(b>UINT64_MAX-a) bad("GGUF addition overflow"); return a+b; }
uint64_t mul(uint64_t a,uint64_t b) { if(a && b>UINT64_MAX/a) bad("GGUF multiplication overflow"); return a*b; }
struct Reader {
    const Mapping& m; uint64_t p=0;
    const unsigned char* take(uint64_t n) { if(p>m.size || n>m.size-p) bad("truncated GGUF"); auto r=m.data+p; p+=n; return r; }
    template<class T> T get() { T v; std::memcpy(&v,take(sizeof(T)),sizeof(T)); return v; }
    std::string str() { auto n=get<uint64_t>(); auto b=take(n); std::string s(reinterpret_cast<const char*>(b),n); if(s.find('\0')!=s.npos) bad("embedded NUL in GGUF string unsupported"); return s; }
    void value(uint32_t t) {
        static constexpr unsigned widths[]={1,1,2,2,4,4,4,1,0,0,8,8,8};
        if(t>=13) bad("unsupported GGUF metadata type");
        if(t==8) { (void)str(); return; }
        if(t==9) { auto sub=get<uint32_t>(); auto n=get<uint64_t>(); if(sub==9 || sub>=13) bad("unsupported nested GGUF array"); if(n>m.size-p) bad("truncated GGUF array"); for(uint64_t i=0;i<n;++i) value(sub); return; }
        auto b=take(widths[t]); if(t==7 && *b>1) bad("invalid GGUF boolean");
    }
};
uint64_t integer(const Metadata& m) {
    if(m.type!=2 && m.type!=4 && m.type!=5 && m.type!=10) bad("invalid GGUF integer metadata type");
    uint64_t v=0; std::memcpy(&v,m.value.data(),m.value.size());
    if(m.type==5 && v>INT32_MAX) bad("negative GGUF split count"); return v;
}
std::pair<uint64_t,uint64_t> format(uint32_t t) {
    switch(t) {
    case 0:return {1,4}; case 1:return {1,2}; case 2:return {32,18}; case 3:return {32,20};
    case 6:return {32,22}; case 7:return {32,24}; case 8:return {32,34};
    case 10:return {256,84}; case 11:return {256,110}; case 12:return {256,144};
    case 13:return {256,176}; case 14:return {256,210}; case 15:return {256,292};
    // gguf 0.19.0 GGML_QUANT_SIZES; repository BLOCK_FORMATS agrees.
    // Storage geometry only: IQ3_S and Q3_K have distinct type identities.
    case 18:return {256,98}; case 20:return {32,18}; case 21:return {256,110};
    case 22:return {256,82}; case 23:return {256,136};
    case 30:return {1,2}; case 42:return {64,18};
    default:bad("unsupported GGML tensor type");
    }
}
}
Mapping::Mapping(const char* path) {
    if(!path || !*path) bad("invalid GGUF path");
    int fd=::open(path,O_RDONLY|O_CLOEXEC); if(fd<0) bad("cannot open GGUF file");
    struct stat st{};
    if(fstat(fd,&st) || !S_ISREG(st.st_mode) || st.st_size<24) { ::close(fd); bad("invalid GGUF file size/type"); }
    size=static_cast<uint64_t>(st.st_size);
    if(size>SIZE_MAX) { ::close(fd); bad("GGUF exceeds address space"); }
    void* ptr=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0); ::close(fd);
    if(ptr==MAP_FAILED) bad("GGUF mmap failed"); data=static_cast<const unsigned char*>(ptr);
}
Mapping::~Mapping() { if(data) munmap(const_cast<unsigned char*>(data),size); }
GGUF::GGUF(const char* const* paths,uint64_t count) {
    if(std::endian::native!=std::endian::little) bad("unsupported host endianness");
    if(!paths || !count || count>65535) bad("invalid GGUF shard count");
    std::set<std::string> names;
    std::map<std::string,std::pair<uint32_t,std::vector<unsigned char>>> signatures;
    std::vector<uint64_t> totals;
    std::set<std::pair<uint64_t,uint64_t>> identities;
    for(uint64_t shard=0;shard<count;++shard) {
        auto owner=std::make_unique<Mapping>(paths[shard]);
        struct stat st{}; if(stat(paths[shard],&st)) bad("cannot stat GGUF shard");
        if(!identities.emplace(st.st_dev,st.st_ino).second) bad("duplicate GGUF shard identity");
        Reader r{*owner}; if(std::memcmp(r.take(4),"GGUF",4)) bad("invalid GGUF magic");
        auto version=r.get<uint32_t>(); if(version!=2 && version!=3) bad("unsupported GGUF version/endianness");
        auto nt=r.get<uint64_t>(), nm=r.get<uint64_t>();
        if(nt>owner->size/32 || nm>owner->size/13) bad("invalid GGUF header counts");
        uint64_t alignment=32; std::map<std::string,size_t> local;
        for(uint64_t j=0;j<nm;++j) {
            Metadata m; m.shard=shard; m.key=r.str(); m.type=r.get<uint32_t>(); auto start=r.p; r.value(m.type);
            m.value.assign(owner->data+start,owner->data+r.p);
            if(!local.emplace(m.key,metadata.size()).second) bad("duplicate GGUF metadata key");
            if(m.key=="general.alignment") { if(m.type!=4) bad("invalid GGUF alignment type"); alignment=integer(m); }
            if(m.key!="split.no") {
                auto sig=std::make_pair(m.type,m.value); auto [it,inserted]=signatures.emplace(m.key,sig);
                if(!inserted && it->second!=sig) bad("inconsistent overlapping GGUF metadata");
            }
            metadata.push_back(std::move(m));
        }
        if(!alignment || (alignment&(alignment-1))) bad("invalid GGUF alignment");
        bool split=local.count("split.no") || local.count("split.count") || local.count("split.tensors.count");
        if(split || count>1) {
            if(!local.count("split.no") || !local.count("split.count") || !local.count("split.tensors.count")) bad("missing GGUF split metadata");
            if(integer(metadata[local.at("split.no")])!=shard || integer(metadata[local.at("split.count")])!=count) bad("invalid GGUF split identity");
            totals.push_back(integer(metadata[local.at("split.tensors.count")]));
        }
        auto first=tensors.size();
        for(uint64_t j=0;j<nt;++j) {
            Tensor t; t.name=r.str(); if(!names.insert(t.name).second) bad("duplicate GGUF tensor name");
            auto& d=t.desc; d.shard_index=shard; d.rank=r.get<uint32_t>();
            if(d.rank<1 || d.rank>4) bad("invalid GGUF tensor rank");
            for(uint64_t k=0;k<d.rank;++k) { d.ggml_shape[k]=r.get<uint64_t>(); if(!d.ggml_shape[k]) bad("invalid GGUF tensor dimension"); }
            d.type_id=r.get<uint32_t>(); auto [elements,bytes]=format(d.type_id); d.block_elements=elements; d.block_bytes=bytes;
            if(d.ggml_shape[0]%elements) bad("invalid GGUF quantized row alignment");
            d.row_bytes=mul(d.ggml_shape[0]/elements,bytes); d.nbytes=d.row_bytes;
            for(uint64_t k=1;k<d.rank;++k) d.nbytes=mul(d.nbytes,d.ggml_shape[k]);
            d.relative_offset=r.get<uint64_t>(); if(d.relative_offset%alignment) bad("unaligned GGUF tensor offset");
            tensors.push_back(std::move(t));
        }
        auto base=add(r.p,alignment-1)&~(alignment-1); if(base>owner->size) bad("truncated GGUF alignment padding");
        std::vector<std::pair<uint64_t,uint64_t>> ranges;
        for(size_t j=first;j<tensors.size();++j) {
            auto& d=tensors[j].desc; d.data_offset=add(base,d.relative_offset); auto end=add(d.data_offset,d.nbytes);
            if(end>owner->size) bad("truncated GGUF tensor range"); ranges.emplace_back(d.data_offset,end);
        }
        std::sort(ranges.begin(),ranges.end());
        for(size_t j=1;j<ranges.size();++j) if(ranges[j].first<ranges[j-1].second) bad("overlapping GGUF tensor ranges");
        maps.push_back(std::move(owner));
    }
    for(auto total:totals) if(total!=tensors.size()) bad("GGUF split tensor total mismatch");
}
}
