cmake_minimum_required(VERSION 3.24)
project(mini_llm_native LANGUAGES CXX CUDA)
find_package(CUDAToolkit 13.0 REQUIRED)
add_library(mini_llm_native SHARED runtime.cpp gguf_loader.cpp qsa_selection.cu qsa_attention.cu qsa_projection.cu qsa_frontend.cu qsa_layer.cu hyperconnection.cu recurrent.cu ple.cu ple_hash.cpp moe.cu)
target_compile_features(mini_llm_native PRIVATE cxx_std_20)
set_target_properties(mini_llm_native PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(mini_llm_native PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
target_compile_options(mini_llm_native PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false;--prec-div=true;--prec-sqrt=true>)
