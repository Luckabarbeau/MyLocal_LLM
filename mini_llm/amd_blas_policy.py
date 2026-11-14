"""Environment-only AMD BLAS preference, applied before importing CuPy."""

import logging
import os


def apply_amd_blas_policy():
    """Apply the process-start preference; never call during a workload.

    ROCBLAS_USE_HIPBLASLT is ROCm-specific and has no effect on CUDA dispatch.
    In auto mode, leave even an externally supplied raw value unchanged.
    """
    policy = os.environ.get("MINI_LLM_AMD_BLAS", "auto")
    if policy not in {"auto", "tensile", "hipblaslt"}:
        raise ValueError(
            "MINI_LLM_AMD_BLAS must be 'auto', 'tensile', or 'hipblaslt'; "
            f"got {policy!r}."
        )
    if policy == "auto":
        return
    effective = "0" if policy == "tensile" else "1"
    os.environ["ROCBLAS_USE_HIPBLASLT"] = effective
    logging.getLogger(__name__).info(
        "MINI_LLM_AMD_BLAS=%s: ROCBLAS_USE_HIPBLASLT=%s",
        policy, effective,
    )
