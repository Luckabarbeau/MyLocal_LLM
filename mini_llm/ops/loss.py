from ..backend import xp, scalar, is_low_precision_dtype, is_bfloat16_dtype, BACKEND_NAME
import os

import numpy as np


_FUSED_BF16_CE_MODULE = None
_FUSED_BF16_CE_DISABLED = False


def _fused_bf16_ce_enabled(dtype):
    raw = os.environ.get("MINI_LLM_FUSED_BF16_CE", "0").strip().lower()
    return (
        BACKEND_NAME == "cupy"
        and is_bfloat16_dtype(dtype)
        and raw not in {"0", "false", "off", "no"}
    )


def _inplace_bf16_ce_enabled(dtype):
    """Reuse the BF16 logits allocation as the CE probability/gradient cache.

    Training never needs logits after cross-entropy forward.  Reusing that
    one-gigabyte-class buffer avoids allocating a second full-vocabulary BF16
    tensor.  It is opt-in because callers that inspect logits after loss
    evaluation may rely on the historical non-mutating API.
    """
    raw = os.environ.get("MINI_LLM_INPLACE_BF16_CE", "0").strip().lower()
    return is_bfloat16_dtype(dtype) and raw not in {"0", "false", "off", "no"}


def _get_fused_bf16_ce_module():
    """Compile the BF16 cross-entropy CUDA kernels lazily.

    The kernels intentionally avoid CUDA headers. BF16 values are read/written
    through their 16-bit IEEE storage representation, while max/sum/log and
    gradient arithmetic remain FP32.
    """
    global _FUSED_BF16_CE_MODULE, _FUSED_BF16_CE_DISABLED
    if BACKEND_NAME != "cupy" or _FUSED_BF16_CE_DISABLED:
        return None
    if _FUSED_BF16_CE_MODULE is not None:
        return _FUSED_BF16_CE_MODULE

    code = r"""
    __device__ __forceinline__ float bf16_to_float(unsigned short x) {
        union { unsigned int u; float f; } v;
        v.u = ((unsigned int)x) << 16;
        return v.f;
    }

    __device__ __forceinline__ unsigned short float_to_bf16(float x) {
        union { unsigned int u; float f; } v;
        v.f = x;
        unsigned int bits = v.u;
        // Round-to-nearest-even before truncating the low 16 mantissa bits.
        unsigned int lsb = (bits >> 16) & 1u;
        bits += 0x7fffu + lsb;
        return (unsigned short)(bits >> 16);
    }

    extern "C" __global__
    void bf16_cross_entropy_fwd(
        const unsigned short* logits,
        const int* targets,
        const float* loss_mask,
        unsigned short* probs,
        float* work,
        float* row_losses,
        int rows,
        int vocab,
        int has_mask) {
        int row = blockIdx.x;
        if (row >= rows) return;
        const unsigned short* z = logits + ((long long)row) * vocab;
        unsigned short* p = probs + ((long long)row) * vocab;
        float* w = work + ((long long)row) * vocab;
        extern __shared__ float sh[];

        float local_max = -3.402823466e+38F;
        for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
            float value = bf16_to_float(z[j]);
            local_max = fmaxf(local_max, value);
        }
        sh[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + stride]);
            __syncthreads();
        }
        float row_max = sh[0];

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
            float e = expf(bf16_to_float(z[j]) - row_max);
            w[j] = e;
            local_sum += e;
        }
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        float row_sum = sh[0];
        float inv_sum = 1.0f / row_sum;

        // Preserve the target logit before writing probabilities.  This barrier
        // is required when 0055A aliases probs with logits in-place: another
        // thread may own the target column and overwrite it below.
        float target_logit = 0.0f;
        if (threadIdx.x == 0)
            target_logit = bf16_to_float(z[targets[row]]);
        __syncthreads();

        for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
            p[j] = float_to_bf16(w[j] * inv_sum);
        }

        if (threadIdx.x == 0) {
            float loss = logf(row_sum) + row_max - target_logit;
            if (has_mask) loss *= loss_mask[row];
            row_losses[row] = loss;
        }
    }

    extern "C" __global__
    void bf16_cross_entropy_bwd(
        unsigned short* probs_and_grad,
        const int* targets,
        const float* loss_mask,
        int rows,
        int vocab,
        float inv_normalizer,
        int has_mask) {
        long long total = ((long long)rows) * vocab;
        long long idx = ((long long)blockIdx.x) * blockDim.x + threadIdx.x;
        long long stride = ((long long)gridDim.x) * blockDim.x;
        for (; idx < total; idx += stride) {
            int row = (int)(idx / vocab);
            int col = (int)(idx - ((long long)row) * vocab);
            float g = bf16_to_float(probs_and_grad[idx]);
            if (col == targets[row]) g -= 1.0f;
            if (has_mask) g *= loss_mask[row];
            g *= inv_normalizer;
            probs_and_grad[idx] = float_to_bf16(g);
        }
    }

    // 0056 loss-only forward for a transient LM-head tile.  No persistent
    // probability tensor or FP32 exp workspace is needed: exp() is recomputed
    // after the row reduction rather than stored.
    extern "C" __global__
    void bf16_cross_entropy_loss_only(
        const unsigned short* logits,
        const int* targets,
        const float* loss_mask,
        float* row_losses,
        int rows,
        int vocab,
        int has_mask) {
        int row = blockIdx.x;
        if (row >= rows) return;
        const unsigned short* z = logits + ((long long)row) * vocab;
        extern __shared__ float sh[];

        float local_max = -3.402823466e+38F;
        for (int j = threadIdx.x; j < vocab; j += blockDim.x)
            local_max = fmaxf(local_max, bf16_to_float(z[j]));
        sh[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + stride]);
            __syncthreads();
        }
        float row_max = sh[0];

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < vocab; j += blockDim.x)
            local_sum += expf(bf16_to_float(z[j]) - row_max);
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            float target_logit = bf16_to_float(z[targets[row]]);
            float loss = logf(sh[0]) + row_max - target_logit;
            if (has_mask) loss *= loss_mask[row];
            row_losses[row] = loss;
        }
    }

    // 0056 recompute-backward kernel.  It converts the transient logits tile
    // directly into dL/dlogits in-place.  Probability values are explicitly
    // rounded through BF16 before subtraction/scaling so the arithmetic matches
    // the established persistent-BF16 CE cache as closely as possible.
    extern "C" __global__
    void bf16_cross_entropy_grad_from_logits(
        unsigned short* logits_and_grad,
        const int* targets,
        const float* loss_mask,
        int rows,
        int vocab,
        float inv_normalizer,
        float grad_scale,
        int has_mask) {
        int row = blockIdx.x;
        if (row >= rows) return;
        unsigned short* z = logits_and_grad + ((long long)row) * vocab;
        extern __shared__ float sh[];

        float local_max = -3.402823466e+38F;
        for (int j = threadIdx.x; j < vocab; j += blockDim.x)
            local_max = fmaxf(local_max, bf16_to_float(z[j]));
        sh[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + stride]);
            __syncthreads();
        }
        float row_max = sh[0];

        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < vocab; j += blockDim.x)
            local_sum += expf(bf16_to_float(z[j]) - row_max);
        sh[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        float inv_sum = 1.0f / sh[0];
        float row_scale = inv_normalizer * grad_scale;
        if (has_mask) row_scale *= loss_mask[row];
        int target = targets[row];

        for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
            float p = expf(bf16_to_float(z[j]) - row_max) * inv_sum;
            // Match the established BF16 probability-cache rounding boundary.
            p = bf16_to_float(float_to_bf16(p));
            if (j == target) p -= 1.0f;
            z[j] = float_to_bf16(p * row_scale);
        }
    }
    """
    try:
        module = xp.RawModule(
            code=code,
            options=("--std=c++11",),
            name_expressions=(
                "bf16_cross_entropy_fwd",
                "bf16_cross_entropy_bwd",
                "bf16_cross_entropy_loss_only",
                "bf16_cross_entropy_grad_from_logits",
            ),
        )
        module.get_function("bf16_cross_entropy_fwd")
        module.get_function("bf16_cross_entropy_bwd")
        module.get_function("bf16_cross_entropy_loss_only")
        module.get_function("bf16_cross_entropy_grad_from_logits")
        _FUSED_BF16_CE_MODULE = module
    except Exception:
        strict = os.environ.get("MINI_LLM_FUSED_BF16_CE_STRICT", "0").strip().lower()
        if strict not in {"0", "false", "off", "no"}:
            raise
        _FUSED_BF16_CE_DISABLED = True
        return None
    return _FUSED_BF16_CE_MODULE


def _fused_bf16_cross_entropy_forward(
    flat_logits, flat_targets, mask_f32, token_chunk, *, reuse_logits=False
):
    module = _get_fused_bf16_ce_module()
    if module is None or not flat_logits.flags.c_contiguous:
        return None

    n, vocab = map(int, flat_logits.shape)
    # Training data targets may still be NumPy host arrays even when logits
    # live on CuPy. NumPy.ndarray.astype(xp.int32) remains a NumPy array, so
    # dtype/contiguity checks alone are not sufficient before RawKernel launch.
    # Force the target indices onto the active backend/device unconditionally.
    targets_i32 = xp.asarray(flat_targets, dtype=xp.int32).reshape(-1)
    if not targets_i32.flags.c_contiguous:
        targets_i32 = xp.ascontiguousarray(targets_i32)
    # 0055A memory mode can safely reuse flat_logits: the kernel finishes all
    # row reads (max + exp workspace) before the synchronized probability write.
    probs_bf16 = flat_logits if reuse_logits else xp.empty(
        flat_logits.shape, dtype=flat_logits.dtype
    )
    row_losses = xp.empty((n,), dtype=xp.float32)
    work_rows = min(int(token_chunk), n)
    work_f32 = xp.empty((work_rows, vocab), dtype=xp.float32)
    if mask_f32 is None:
        mask_arg = xp.empty((1,), dtype=xp.float32)
        has_mask = 0
    else:
        mask_arg = mask_f32
        has_mask = 1

    threads = 256
    kernel = module.get_function("bf16_cross_entropy_fwd")
    for start in range(0, n, token_chunk):
        end = min(start + token_chunk, n)
        rows = end - start
        kernel(
            (rows,), (threads,),
            (
                flat_logits[start:end],
                targets_i32[start:end],
                mask_arg if mask_f32 is None else mask_arg[start:end],
                probs_bf16[start:end],
                work_f32[:rows],
                row_losses[start:end],
                np.int32(rows),
                np.int32(vocab),
                np.int32(has_mask),
            ),
            shared_mem=threads * 4,
        )
    return probs_bf16, targets_i32, row_losses


def _fused_bf16_cross_entropy_backward(probs_bf16, targets_i32, mask_f32, normalizer_count, token_chunk):
    module = _get_fused_bf16_ce_module()
    if module is None:
        return False
    flat = probs_bf16.reshape(-1, probs_bf16.shape[-1])
    n, vocab = map(int, flat.shape)
    if mask_f32 is None:
        mask_arg = xp.empty((1,), dtype=xp.float32)
        has_mask = 0
    else:
        mask_arg = mask_f32
        has_mask = 1
    threads = 256
    kernel = module.get_function("bf16_cross_entropy_bwd")
    inv_normalizer = np.float32(1.0 / float(normalizer_count))
    for start in range(0, n, token_chunk):
        end = min(start + token_chunk, n)
        rows = end - start
        total = rows * vocab
        blocks = min(65535, max(1, (total + threads - 1) // threads))
        kernel(
            (blocks,), (threads,),
            (
                flat[start:end],
                targets_i32[start:end],
                mask_arg if mask_f32 is None else mask_arg[start:end],
                np.int32(rows),
                np.int32(vocab),
                inv_normalizer,
                np.int32(has_mask),
            ),
        )
    return True


def chunked_bf16_cross_entropy_loss(logits, targets, *, loss_mask=None, normalizer_count=None):
    """0056 loss-only CE for one transient BF16 LM-head tile.

    Returns a device FP32 scalar and does not retain probabilities.  The CUDA
    fast path allocates only per-row losses; logits may be discarded immediately
    after this call.  ``normalizer_count`` is the global token/mask normalizer,
    not the local tile size.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    n, vocab = map(int, flat_logits.shape)
    targets_i32 = xp.asarray(targets, dtype=xp.int32).reshape(-1)
    if targets_i32.shape[0] != n:
        raise ValueError("targets must have one entry per logits row")
    if normalizer_count is None:
        normalizer_count = float(n)
    normalizer_count = float(normalizer_count)
    if normalizer_count <= 0.0:
        raise ValueError("normalizer_count must be positive")
    mask_f32 = None if loss_mask is None else xp.asarray(loss_mask, dtype=xp.float32).reshape(-1)
    if mask_f32 is not None and mask_f32.shape[0] != n:
        raise ValueError("loss_mask must have one entry per logits row")

    module = _get_fused_bf16_ce_module() if _fused_bf16_ce_enabled(logits.dtype) else None
    if module is not None and flat_logits.flags.c_contiguous:
        if not targets_i32.flags.c_contiguous:
            targets_i32 = xp.ascontiguousarray(targets_i32)
        row_losses = xp.empty((n,), dtype=xp.float32)
        if mask_f32 is None:
            mask_arg = xp.empty((1,), dtype=xp.float32)
            has_mask = 0
        else:
            mask_arg = mask_f32
            has_mask = 1
        threads = 256
        module.get_function("bf16_cross_entropy_loss_only")(
            (n,), (threads,),
            (
                flat_logits, targets_i32, mask_arg, row_losses,
                np.int32(n), np.int32(vocab), np.int32(has_mask),
            ),
            shared_mem=threads * 4,
        )
        return xp.sum(row_losses) / normalizer_count

    # Portable/reference fallback.  The tile is deliberately small.
    work = flat_logits.astype(xp.float32, copy=True)
    rows = xp.arange(n)
    max_logit = xp.max(work, axis=-1, keepdims=True)
    work -= max_logit
    target_shifted = work[rows, targets_i32].copy()
    xp.exp(work, out=work)
    denom = xp.sum(work, axis=-1)
    per_token = xp.log(denom) - target_shifted
    if mask_f32 is not None:
        per_token *= mask_f32
    return xp.sum(per_token) / normalizer_count


def chunked_bf16_cross_entropy_grad_inplace(
    logits, targets, *, loss_mask=None, normalizer_count=None, grad_scale=1.0
):
    """0056 overwrite one transient BF16 logits tile with dL/dlogits."""
    flat = logits.reshape(-1, logits.shape[-1])
    n, vocab = map(int, flat.shape)
    targets_i32 = xp.asarray(targets, dtype=xp.int32).reshape(-1)
    if normalizer_count is None:
        normalizer_count = float(n)
    normalizer_count = float(normalizer_count)
    mask_f32 = None if loss_mask is None else xp.asarray(loss_mask, dtype=xp.float32).reshape(-1)

    module = _get_fused_bf16_ce_module() if _fused_bf16_ce_enabled(logits.dtype) else None
    if module is not None and flat.flags.c_contiguous:
        if not targets_i32.flags.c_contiguous:
            targets_i32 = xp.ascontiguousarray(targets_i32)
        if mask_f32 is None:
            mask_arg = xp.empty((1,), dtype=xp.float32)
            has_mask = 0
        else:
            mask_arg = mask_f32
            has_mask = 1
        threads = 256
        module.get_function("bf16_cross_entropy_grad_from_logits")(
            (n,), (threads,),
            (
                flat, targets_i32, mask_arg,
                np.int32(n), np.int32(vocab),
                np.float32(1.0 / normalizer_count), np.float32(grad_scale),
                np.int32(has_mask),
            ),
            shared_mem=threads * 4,
        )
        return flat.reshape(logits.shape)

    # Reference fallback.  Preserve the BF16 probability rounding boundary.
    work = flat.astype(xp.float32, copy=True)
    rows = xp.arange(n)
    max_logit = xp.max(work, axis=-1, keepdims=True)
    work -= max_logit
    xp.exp(work, out=work)
    work /= xp.sum(work, axis=-1, keepdims=True)
    if is_bfloat16_dtype(logits.dtype):
        work = work.astype(logits.dtype).astype(xp.float32)
    work[rows, targets_i32] -= 1.0
    if mask_f32 is not None:
        work *= mask_f32[:, None]
    work *= float(grad_scale) / normalizer_count
    flat[...] = work.astype(logits.dtype)
    return flat.reshape(logits.shape)


def cross_entropy_forward(logits, targets, loss_mask=None, return_device_loss=False):
    """
    Cross entropy loss with mixed precision support.
    
    Performs log-sum-exp in float32 for numerical stability,
    especially important for FP16/BF16 inputs where underflow is common.
    """
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)
    n = flat_logits.shape[0]

    if is_bfloat16_dtype(logits.dtype):
        # BF16 has the same exponent range as FP32, so unlike FP16 it can store
        # tiny softmax probabilities and the 1/N cross-entropy gradient without
        # underflow. Keep all numerically sensitive softmax/log-sum-exp work in
        # FP32, but store the persistent probability cache in BF16. This removes
        # the two full-vocabulary FP32 tensors that otherwise dominate peak VRAM
        # for large vocabularies.
        raw_chunk = os.environ.get("MINI_LLM_LOSS_TOKEN_CHUNK", "512")
        token_chunk = max(1, min(n, int(raw_chunk)))

        if loss_mask is None:
            mask_f32 = None
            normalizer_count = float(n)
        else:
            mask_f32 = xp.asarray(loss_mask, dtype="float32").reshape(-1)
            if mask_f32.shape[0] != n:
                raise ValueError(
                    f"loss_mask has {mask_f32.shape[0]} elements, expected {n}"
                )
            normalizer_count_backend = xp.sum(mask_f32)
            normalizer_count = float(normalizer_count_backend.item())
            if normalizer_count <= 0.0:
                raise ValueError("loss_mask must select at least one target token")

        reuse_logits = _inplace_bf16_ce_enabled(logits.dtype)

        if _fused_bf16_ce_enabled(logits.dtype):
            fused = _fused_bf16_cross_entropy_forward(
                flat_logits, flat_targets, mask_f32, token_chunk,
                reuse_logits=reuse_logits,
            )
            if fused is not None:
                probs_bf16, targets_i32, row_losses = fused
                loss_f32 = xp.sum(row_losses) / normalizer_count
                return (loss_f32 if return_device_loss else scalar(loss_f32)), {
                    "probs_bf16": probs_bf16,
                    "targets": targets_i32,
                    "original_shape": logits.shape,
                    "n": n,
                    "loss_mask_f32": mask_f32,
                    "normalizer_count": normalizer_count,
                    "token_chunk": token_chunk,
                    "fused_bf16_ce": True,
                    "inplace_bf16_ce": reuse_logits,
                }

        probs_bf16 = (
            flat_logits
            if reuse_logits
            else xp.empty(flat_logits.shape, dtype=logits.dtype)
        )
        loss_sum = xp.asarray(0.0, dtype="float32")
        for start in range(0, n, token_chunk):
            end = min(start + token_chunk, n)
            work = flat_logits[start:end].astype("float32", copy=True)
            local_targets = flat_targets[start:end]
            rows = xp.arange(end - start)

            max_logit = xp.max(work, axis=-1, keepdims=True)
            work -= max_logit
            target_shifted = work[rows, local_targets].copy()

            xp.exp(work, out=work)
            normalizer = xp.sum(work, axis=-1, keepdims=True)
            per_token_loss = xp.log(normalizer[:, 0]) - target_shifted
            if mask_f32 is None:
                loss_sum += xp.sum(per_token_loss)
            else:
                loss_sum += xp.sum(per_token_loss * mask_f32[start:end])

            xp.divide(work, normalizer, out=work)
            probs_bf16[start:end] = work.astype(logits.dtype)

        loss_f32 = loss_sum / normalizer_count
        return (loss_f32 if return_device_loss else scalar(loss_f32)), {
            "probs_bf16": probs_bf16,
            "targets": flat_targets,
            "original_shape": logits.shape,
            "n": n,
            "loss_mask_f32": mask_f32,
            "normalizer_count": normalizer_count,
            "token_chunk": token_chunk,
            "inplace_bf16_ce": reuse_logits,
        }

    if is_low_precision_dtype(logits.dtype):
        # FP16 still needs a full FP32 probability cache: its exponent range is
        # too small for tiny non-target probabilities/gradients at large vocab.
        work = flat_logits.astype("float32", copy=True)

        rows = xp.arange(n)
        max_logit = xp.max(work, axis=-1, keepdims=True)
        work -= max_logit
        target_shifted = work[rows, flat_targets].copy()

        xp.exp(work, out=work)
        normalizer = xp.sum(work, axis=-1, keepdims=True)
        per_token_loss = xp.log(normalizer[:, 0]) - target_shifted
        if loss_mask is None:
            loss_f32 = xp.mean(per_token_loss)
            mask_f32 = None
            normalizer_count = float(n)
        else:
            mask_f32 = xp.asarray(loss_mask, dtype="float32").reshape(-1)
            if mask_f32.shape[0] != n:
                raise ValueError(
                    f"loss_mask has {mask_f32.shape[0]} elements, expected {n}"
                )
            normalizer_count_backend = xp.sum(mask_f32)
            normalizer_count = float(normalizer_count_backend.item())
            if normalizer_count <= 0.0:
                raise ValueError("loss_mask must select at least one target token")
            loss_f32 = xp.sum(per_token_loss * mask_f32) / normalizer_count

        xp.divide(work, normalizer, out=work)
        cache = {
            "probs_f32": work,
            "targets": flat_targets,
            "original_shape": logits.shape,
            "n": n,
            "loss_mask_f32": mask_f32,
            "normalizer_count": normalizer_count,
        }
        return (loss_f32 if return_device_loss else scalar(loss_f32)), cache
    else:
        # Float32 path - standard computation
        max_logit = xp.max(flat_logits, axis=-1, keepdims=True)
        shifted = flat_logits - max_logit
        exp_logits = xp.exp(shifted)
        probs = exp_logits / xp.sum(exp_logits, axis=-1, keepdims=True)

        rows = xp.arange(n)
        target_probs = probs[rows, flat_targets]
        per_token_loss = -xp.log(target_probs + 1e-30)
        if loss_mask is None:
            loss = xp.mean(per_token_loss)
            mask_f32 = None
            normalizer_count = float(n)
        else:
            mask_f32 = xp.asarray(loss_mask, dtype="float32").reshape(-1)
            if mask_f32.shape[0] != n:
                raise ValueError(
                    f"loss_mask has {mask_f32.shape[0]} elements, expected {n}"
                )
            normalizer_count_backend = xp.sum(mask_f32)
            normalizer_count = float(normalizer_count_backend.item())
            if normalizer_count <= 0.0:
                raise ValueError("loss_mask must select at least one target token")
            loss = xp.sum(per_token_loss * mask_f32) / normalizer_count

        return (loss if return_device_loss else scalar(loss)), {
            "probs": probs,
            "targets": flat_targets,
            "original_shape": logits.shape,
            "n": n,
            "loss_mask_f32": mask_f32,
            "normalizer_count": normalizer_count,
        }


def cross_entropy_backward(cache):
    """
    Backward pass for cross entropy loss.
    
    For FP16 logits, returns the loss gradient in FP32. Keeping dL/dlogits
    in FP32 is important because the 1/N cross-entropy scaling can push many
    non-target components below the representable FP16 range before loss
    scaling is applied by the trainer.
    """
    probs_bf16 = cache.pop("probs_bf16", None)
    targets = cache["targets"]
    n = cache["n"]
    original_shape = cache["original_shape"]
    loss_mask_f32 = cache.get("loss_mask_f32")
    normalizer_count = float(cache.get("normalizer_count", n))

    if probs_bf16 is not None:
        # Reuse the BF16 probability buffer itself as d_logits.
        token_chunk = int(cache.get("token_chunk", 512))
        if cache.get("fused_bf16_ce", False):
            if _fused_bf16_cross_entropy_backward(
                probs_bf16, targets, loss_mask_f32, normalizer_count, token_chunk
            ):
                return probs_bf16.reshape(original_shape)

        # Vectorized fallback: promote each token chunk to FP32 for target
        # subtraction, masking and normalization, then store back in BF16.
        flat = probs_bf16.reshape(n, -1)
        for start in range(0, n, token_chunk):
            end = min(start + token_chunk, n)
            work = flat[start:end].astype("float32", copy=True)
            rows = xp.arange(end - start)
            work[rows, targets[start:end]] -= 1.0
            if loss_mask_f32 is not None:
                work *= loss_mask_f32[start:end, None]
            work /= normalizer_count
            flat[start:end] = work.astype(probs_bf16.dtype)
        return probs_bf16.reshape(original_shape)

    probs_f32 = cache.get("probs_f32", None)
    if probs_f32 is not None:
        probs = probs_f32.copy()
    else:
        probs = cache["probs"].copy()

    rows = xp.arange(n)
    probs[rows, targets] -= 1.0
    if loss_mask_f32 is not None:
        probs *= loss_mask_f32[:, None]
    probs /= normalizer_count
    return probs.reshape(original_shape)
