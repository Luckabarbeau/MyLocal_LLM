"""0106K matched-storage FP64 arithmetic oracle for small packed fixtures.

This deliberately does not call the production linear or activation operators.
Packed decoding is shared with dequantize_reference(), so this diagnoses
arithmetic precision, not independently the GGUF bit layout. Never use it to
materialize a full external model or as a production inference fallback.
"""

import numpy as np

from mini_llm.ops.quantized_weight import PackedGGUFWeight


def packed_swiglu_reference(x, gate, up, down=None, *, storage_dtype=None):
    """Return FP64 stages and explicitly rounded activation-storage boundaries.

    Graph: round(x) -> FP64 gate/up -> FP64 SiLU(gate)*up -> round(hidden)
           -> optional FP64 down -> round(output).

    A fused gate/up kernel does NOT store FP16 intermediate projections: only
    input, hidden and final output use storage_dtype. The returned gate/up and
    unrounded stages remain FP64. Actual device inputs should either be passed
    already rounded or specify the same storage_dtype. Defaults to x.dtype.
    """
    x = np.asarray(x)
    if x.dtype.kind != "f" or x.ndim not in (2, 3):
        raise ValueError("Oracle input must be floating-point [M,K] or [B,T,K]")
    dtype = np.dtype(x.dtype if storage_dtype is None else storage_dtype)
    if dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
        raise TypeError("Oracle storage dtype must be float16, float32 or float64")
    if not np.isfinite(x).all():
        raise ValueError("Oracle input must be finite")
    projections = (gate, up) if down is None else (gate, up, down)
    if not all(isinstance(w, PackedGGUFWeight) and len(w.shape) == 2
               for w in projections):
        raise TypeError("Oracle weights must be host-packed two-dimensional matrices")
    if gate.shape != up.shape or gate.shape[1] != x.shape[-1]:
        raise ValueError("Oracle gate/up and input dimensions must match")
    if down is not None and down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError("Oracle down dimensions must be transposed gate dimensions")

    effective = x.astype(dtype).astype(np.float64)
    matrices = [w.dequantize_reference().astype(np.float64) for w in projections]
    g = effective @ matrices[0].T
    u = effective @ matrices[1].T
    # Stable on large negative values as well as the cancellation-heavy tests.
    exp_neg_abs = np.exp(-np.abs(g))
    sigmoid = np.where(g >= 0, 1.0 / (1.0 + exp_neg_abs),
                       exp_neg_abs / (1.0 + exp_neg_abs))
    hidden_unrounded = (g * sigmoid) * u
    hidden = hidden_unrounded.astype(dtype)
    if down is None:
        output_unrounded, output = hidden_unrounded, hidden
    else:
        output_unrounded = hidden.astype(np.float64) @ matrices[2].T
        output = output_unrounded.astype(dtype)
    return dict(input=effective, gate=g, up=u,
                hidden_unrounded=hidden_unrounded, hidden=hidden,
                output_unrounded=output_unrounded, output=output)
