"""Compatibility adapters for the HTTP service.

Model specs belong to model_sources and all weight loading belongs to
model_loading. ServingModel only adapts token/state calls to the existing
in-process inference engines. Remote models use completion-level adapters.
"""

from .model_sources import ModelSpec, SamplingConfig, load_specs  # public compat reexports



class ServingModel:
    def __init__(self, engine, tokenizer, context_length, *, native=False, stop_ids=(),
                 execution_policy="auto"):
        self.engine = engine
        self.tokenizer = tokenizer
        self.context_length = int(context_length)
        self.native = native
        self.stop_ids = set(stop_ids)
        self.execution_policy = execution_policy

    def encode(self, text):
        if self.native:
            return list(self.tokenizer.encode(text))
        return list(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, ids):
        if self.native:
            return self.tokenizer.decode(ids)
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def create_state(self, total_tokens):
        return self.engine.create_generation_state(1, total_tokens)

    def prefill(self, ids, state):
        from mini_llm.backend import xp
        logits = self.engine.prefill(xp.asarray([ids], dtype=xp.int32), state)
        from .inference_policy import observe_execution
        native_engine = getattr(self.engine, "engine", self.engine)
        report = observe_execution(native_engine, state)
        state.execution_report = report
        # Strict-fast is validated *after* decoding, when a graph has had an
        # opportunity to execute. Checking here mistakes capture for execution.
        return logits

    def execution_report(self, state):
        """Observe the *completed* request, not only post-prefill capture.

        CUDA graphs may first execute during decode_one. A report saved in
        prefill can therefore be stale even when the kernel ran correctly.
        """
        from .inference_policy import observe_execution
        native_engine = getattr(self.engine, "engine", self.engine)
        return observe_execution(native_engine, state)

    def decode_one(self, token_id, state):
        from mini_llm.backend import xp
        return self.engine.decode_one(xp.asarray([token_id], dtype=xp.int32), state)


def _external_stops(tokenizer):
    return {int(i) for i in (
        tokenizer.token_to_id(t) for t in
        ("<|im_end|>", "<|endoftext|>", "</s>", "<|eos|>")) if i is not None}


def _native_stops(tokenizer):
    return {int(i) for t in ("<|eos|>", "</s>") for ids in [tokenizer.encode(t)]
            if len(ids) == 1 for i in ids}


def load_model(spec):
    from .model_loading import load_native_checkpoint, load_pretrained_weights
    fmt = spec.resolved_format()
    if fmt == "native":
        from .inference_model import InferenceModel
        execution_policy = getattr(spec, "execution_policy", "auto")
        loaded = load_native_checkpoint(spec.path,
                                        apply_profile=execution_policy != "reference",
                                        execution_policy=execution_policy)
        engine = InferenceModel(loaded.model.config, dtype=loaded.model.dtype)
        engine.set_weights(loaded.model)
        tokenizer = loaded.tokenizer
        cap = loaded.context_length
        native = True
    elif fmt in {"gguf", "safetensors"}:
        from .backend import BACKEND_NAME
        from .inference_policy import configure_inference
        configure_inference(None, backend=BACKEND_NAME, dtype=spec.dtype,
                            policy=getattr(spec, "execution_policy", "auto"), model_format=fmt)
        loaded = load_pretrained_weights(spec.path, format=fmt, dtype=spec.dtype)
        engine, tokenizer, cap = loaded.engine, loaded.tokenizer, loaded.context_length
        native = False
    elif fmt == "remote":
        from .remote_provider import RemoteCompletionEngine
        return RemoteCompletionEngine(spec)
    else:
        raise ValueError(f"Unknown model format: {fmt}")
    if spec.context_length is not None:
        if int(spec.context_length) > cap:
            raise ValueError(f"Configured context {spec.context_length} exceeds model capacity {cap}")
        cap = int(spec.context_length)
    stops = _native_stops(tokenizer) if native else _external_stops(tokenizer)
    return ServingModel(engine, tokenizer, cap, native=native, stop_ids=stops,
                        execution_policy=getattr(spec, "execution_policy", "auto"))
