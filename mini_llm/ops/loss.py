from ..backend import xp, scalar, is_low_precision_dtype, is_bfloat16_dtype
import os


def cross_entropy_forward(logits, targets, loss_mask=None):
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
        probs_bf16 = xp.empty(flat_logits.shape, dtype=logits.dtype)

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
        return scalar(loss_f32), {
            "probs_bf16": probs_bf16,
            "targets": flat_targets,
            "original_shape": logits.shape,
            "n": n,
            "loss_mask_f32": mask_f32,
            "normalizer_count": normalizer_count,
            "token_chunk": token_chunk,
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
        return scalar(loss_f32), cache
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

        return scalar(loss), {
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
        # Reuse the BF16 probability buffer itself as d_logits.  Each token
        # chunk is promoted to FP32 for target subtraction, masking and 1/N
        # normalization, then stored back in BF16. DecoderLanguageModel already
        # feeds BF16 d_logits to the large output GEMM, so this removes a full
        # FP32 d_logits allocation without changing the compute dtype of that GEMM.
        flat = probs_bf16.reshape(n, -1)
        token_chunk = int(cache.get("token_chunk", 512))
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
