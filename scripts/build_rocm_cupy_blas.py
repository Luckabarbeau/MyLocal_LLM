#!/usr/bin/env python3
"""Build only the patched CuPy 14.2.0 BLAS binding; does not install it.

Run with .venv-rocm's Python and CUPY_SOURCE pointing to matching CuPy sources.
The build runs in an isolated directory to avoid CuPy packaging metadata.
The source must contain the CUDA_R_16BF -> HIPBLAS_R_16B mapping.
"""
import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

source = Path(os.environ["CUPY_SOURCE"]).resolve()
build_dir = source.parent / "cupy-bf16-build"
build_dir.mkdir(exist_ok=True)
os.chdir(build_dir)
rocm = Path(os.environ.get("ROCM_HOME", "/opt/rocm-7.2.4"))
header = source / "cupy_backends/hip/cupy_hipblas.h"
if "case 14 /* CUDA_R_16BF */: return HIPBLAS_R_16B;" not in header.read_text():
    raise RuntimeError("Apply the BF16 datatype mapping patch before building")

extension = Extension(
    "cupy_backends.cuda.libs.cublas",
    [str(source / "cupy_backends/cuda/libs/cublas.pyx")],
    language="c++",
    include_dirs=[str(source), str(source / "cupy/_core/include"),
                  str(rocm / "include"), str(rocm / "include/hipblas"),
                  str(rocm / "include/rocsolver"), str(rocm / "include/rocblas")],
    library_dirs=[str(rocm / "lib")],
    libraries=["hipblas", "amdhip64"],
    define_macros=[("CUPY_USE_HIP", "1"), ("__HIP_PLATFORM_AMD__", "1"),
                   ("__HIP_PLATFORM_HCC__", "1")],
)
setup(
    name="cupy-rocm-bf16-blas-fix",
    packages=[],
    ext_modules=cythonize(
        [extension], include_path=[str(source)],
        compile_time_env={"CUPY_USE_CUDA_PYTHON": False,
                                  "CUPY_CUDA_VERSION": 0, "CUPY_HIP_VERSION": 70200000},
        compiler_directives={"language_level": 3},
    ),
)
