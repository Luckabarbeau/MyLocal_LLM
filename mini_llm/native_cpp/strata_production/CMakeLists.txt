cmake_minimum_required(VERSION 3.24)
project(mini_llm_strata_production LANGUAGES NONE)
include(ExternalProject)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(STRATA_PIN 712fec6b4297c9cfd32ab43daaa2979a9aa01edf)
get_filename_component(STRATA_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/strata-712fec6b4297" ABSOLUTE)
set(STRATA_BUILD_JOBS 4 CACHE STRING "Maximum upstream compiler processes")
set(STRATA_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE FILEPATH "CUDA 13+ nvcc")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/verify_source.py"
  COMMAND_ERROR_IS_FATAL ANY)

# Keep upstream's target graph and per-target release flags in a separate CMake scope.
ExternalProject_Add(strata_upstream
  SOURCE_DIR "${STRATA_SOURCE}"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/upstream"
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CUDA_COMPILER:FILEPATH=${STRATA_CUDA_COMPILER}
    -DCMAKE_CUDA_ARCHITECTURES:STRING=120
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DSTRATA_ENABLE_CUDA=ON
    -DSTRATA_BUILD_TESTS=OFF
    -DSTRATA_BUILD_CONVERSATION_TESTS=OFF
    -DSTRATA_NATIVE_EXPERTS=ON
    -DSTRATA_PORTABLE=ON
    -DSTRATA_Q6K_EXPERTS=ON
    -DSTRATA_MMQ_KQUANTS=ON
    -DSTRATA_MMQ_TQ_OVERLAY=ON

    -DSTRATA_GGML_DIR:PATH=
  BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target strata strata_kq --parallel ${STRATA_BUILD_JOBS}
  BUILD_ALWAYS TRUE
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/upstream/strata" "${CMAKE_CURRENT_BINARY_DIR}/upstream/libstrata_kq.so")
add_custom_target(mini_llm_strata_engine DEPENDS strata_upstream)
