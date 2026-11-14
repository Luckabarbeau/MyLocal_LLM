#!/usr/bin/env python3
"""Opt-in native CuPy/HIP + CuPy/CUDA synchronous, elastic accumulation.

The coordinator never imports a GPU runtime. Bulk tensors use local host-backed
mmap files; inherited Unix sockets carry only control messages. No collectives,
PyTorch, or cross-vendor device pointers are used.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parent


class Channel:
    def __init__(self, sock):
        self.sock = sock
        self.buffer = bytearray()

    def send(self, **message):
        self.sock.sendall(json.dumps(message, allow_nan=False).encode() + b"\n")

    def receive(self, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.sock], [], [], remaining)[0]:
                raise TimeoutError("worker control channel timed out")
            data = self.sock.recv(65536)
            if not data:
                raise RuntimeError("worker exited or closed its control channel; inspect worker log")
            self.buffer.extend(data)
        line, _, rest = self.buffer.partition(b"\n")
        self.buffer = bytearray(rest)
        message = json.loads(line)
        if message.get("event") == "error":
            raise RuntimeError(message["message"])
        return message


def validate_config(config):
    workers = config.get("workers", [])
    if len(workers) < 2:
        raise ValueError("at least two workers are required")
    names = [w["name"] for w in workers]
    if len(set(names)) != len(names):
        raise ValueError("worker names must be unique")
    if config.get("anchor", names[0]) not in names:
        raise ValueError("anchor must name a worker")
    for worker in workers:
        for key in ("batch_size", "context_length"):
            if type(worker.get(key)) is not int or worker[key] < 1:
                raise ValueError(f"{worker['name']}: {key} must be a positive integer")
        if worker.get("backend", "cupy") != "cupy" and not config.get("smoke"):
            raise ValueError("production workers must use native CuPy")
        if worker.get("expected_runtime") not in {None, "cuda", "hip"}:
            raise ValueError("expected_runtime must be cuda or hip")
    anchor = workers[names.index(config.get("anchor", names[0]))]
    if type(anchor.get("target_microbatches")) is not int or anchor["target_microbatches"] < 1:
        raise ValueError("anchor needs positive target_microbatches")
    if not math.isfinite(float(config.get("timeout_s", 1800))) or float(config.get("timeout_s", 1800)) <= 0:
        raise ValueError("timeout_s must be finite and positive")
    warmup = config.get("warmup_microbatches", 1)
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup_microbatches must be a nonnegative integer")
    fraction = config.get("target_compute_fraction")
    if fraction is not None and (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                                 or not math.isfinite(fraction) or not 0 < fraction < 1):
        raise ValueError("target_compute_fraction must be finite and in (0,1)")
    maximum = config.get("max_anchor_microbatches", anchor["target_microbatches"])
    if type(maximum) is not int or maximum < anchor["target_microbatches"]:
        raise ValueError("max_anchor_microbatches must be an integer >= anchor target")
    if fraction is not None and "max_anchor_microbatches" not in config:
        raise ValueError("adaptive accumulation requires explicit max_anchor_microbatches")
    for worker in workers:
        cap = worker.get("max_microbatches")
        if cap is not None and (type(cap) is not int or cap < 1):
            raise ValueError("max_microbatches must be a positive integer")
        if worker is anchor and cap is not None and cap < maximum:
            raise ValueError("anchor max_microbatches must cover the maximum anchor target")
    if config.get("init_from") and not config.get("acknowledge_nonexact_init"): 
        raise ValueError("init_from is weights-only: acknowledge_nonexact_init must be true")
    if not config.get("smoke") and (not config.get("train_shards") or "eos_token_id" not in config):
        raise ValueError("production requires explicit train_shards and eos_token_id")


def next_anchor_target(base_target, max_target, microbatch_s, barrier_s, compute_fraction):
    """Amortize the measured update barrier; never exceed the configured budget."""
    if compute_fraction is None or microbatch_s <= 0 or barrier_s <= 0:
        return base_target
    desired_compute_s = barrier_s * compute_fraction / (1.0 - compute_fraction)
    return max(base_target, min(max_target, math.ceil(desired_compute_s / microbatch_s)))


def reduce_gradients(sources, destination, supervised_tokens, chunk_elements=1048576):
    """Sum unnormalised loss gradients, then normalise ONCE by global weight."""
    if not math.isfinite(supervised_tokens) or supervised_tokens <= 0:
        raise ValueError("global supervised-token count must be finite and positive")
    if not sources or any(source.shape != destination.shape for source in sources):
        raise ValueError("gradient buffer shape mismatch")
    for start in range(0, destination.size, chunk_elements):
        stop = min(destination.size, start + chunk_elements)
        out = destination[start:stop]
        out.fill(0)
        for source in sources:
            part = source[start:stop]
            if not np.isfinite(part).all():
                raise ValueError("nonfinite worker gradient; update aborted")
            out += part
        out /= supervised_tokens
        if not np.isfinite(out).all():
            raise ValueError("nonfinite aggregate gradient; update aborted")


def run(config, *, steps=1, work_dir=None):
    validate_config(config)
    if steps < 1:
        raise ValueError("steps must be positive")
    workers = config["workers"]
    anchor = [w["name"] for w in workers].index(config.get("anchor", workers[0]["name"]))
    base_target = workers[anchor]["target_microbatches"]
    target = base_target
    maximum_target = config.get("max_anchor_microbatches", base_target)
    anchor_microbatch_ema = barrier_ema = None
    timeout = float(config.get("timeout_s", 1800))
    # Worker zero owns all FP32 optimizer masters/moments. Others hold no optimizer.
    processes, channels, logs = [], [], []
    results = []
    with tempfile.TemporaryDirectory(prefix="native-heterogeneous-", dir=work_dir) as temporary:
        directory = Path(temporary)
        config_path = directory / "run.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        try:
            for index, worker in enumerate(workers):
                parent, child = socket.socketpair()
                env = os.environ.copy()
                # Do not leak a restored production/CUDA profile into a HIP worker.
                for key in list(env):
                    if key.startswith("MINI_LLM_"):
                        del env[key]
                env.update(config.get("env", {}))
                env.update(worker.get("env", {}))
                env["MINI_LLM_BACKEND"] = worker.get("backend", "cupy")
                env.setdefault("MINI_LLM_OPTIMIZER_OFFLOAD", "full")
                interpreter = str((ROOT / worker["python"]).absolute())
                log_path = Path(config.get("log_dir", ROOT / "logs"))
                log_path.mkdir(parents=True, exist_ok=True)
                log = (log_path / f"heterogeneous-worker-{index}.log").open("w")
                logs.append(log)
                try:
                    process = subprocess.Popen(
                        [interpreter, "-u", str(Path(__file__).resolve()), "--worker-fd", str(child.fileno()),
                         "--worker-index", str(index), "--config", str(config_path)],
                        cwd=ROOT, env=env, pass_fds=(child.fileno(),), stdout=log, stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                processes.append(process)
                channels.append(Channel(parent))
            ready = [channel.receive(timeout) for channel in channels]
            schema = ready[0]["schema"]
            if any(message.get("event") != "ready" or message["schema"] != schema for message in ready):
                raise ValueError("worker parameter names/shapes disagree")
            count = sum(math.prod(shape) for _, shape in schema)
            buffers = {}
            for name in ["weights", "aggregate"] + [f"grad-{i}" for i in range(len(workers))]:
                buffers[name] = np.memmap(directory / name, mode="w+", dtype=np.float32, shape=(count,))
            for channel in channels:
                channel.send(command="map", directory=str(directory), count=count)
            for channel in channels:
                assert channel.receive(timeout)["event"] == "mapped"
            channels[0].send(command="publish")
            assert channels[0].receive(timeout)["event"] == "published"
            for channel in channels:
                channel.send(command="sync", initialize_optimizer=True)
            for channel in channels:
                assert channel.receive(timeout)["event"] == "synced"
            warmup = {}
            warmup_count = config.get("warmup_microbatches", 1)
            if warmup_count:
                for channel in channels:
                    channel.send(command="warmup", microbatches=warmup_count)
                for worker, channel in zip(workers, channels):
                    report = channel.receive(timeout)
                    if report["event"] != "warmed":
                        raise RuntimeError(f"unexpected warmup response: {report}")
                    warmup[worker["name"]] = report
            for step in range(1, steps + 1):
                started = time.monotonic()
                round_target = target
                for index, channel in enumerate(channels):
                    channel.send(command="begin", step=step,
                                 target_microbatches=round_target if index == anchor else None)
                reports = {}
                closing = False
                deadline = time.monotonic() + timeout
                while len(reports) < len(workers):
                    pending = [i for i in range(len(workers)) if i not in reports]
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("accumulation round timed out")
                    readable = [channels[i].sock for i in pending if b"\n" in channels[i].buffer]
                    if not readable:
                        readable, _, _ = select.select([channels[i].sock for i in pending], [], [], remaining)
                    if not readable:
                        raise TimeoutError("accumulation round timed out")
                    for index in sorted(pending, key=lambda i: i != anchor):
                        channel = channels[index]
                        if channel.sock not in readable and b"\n" not in channel.buffer:
                            continue
                        report = channel.receive(max(0.001, deadline - time.monotonic()))
                        if report["event"] == "accumulated" and index == anchor and not closing:
                            closing = True
                            for peer, peer_channel in enumerate(channels):
                                if peer != anchor:
                                    peer_channel.send(command="finish")
                        elif report["event"] == "exported":
                            reports[index] = report
                        else:
                            raise RuntimeError(f"unexpected worker event: {report}")
                accumulation_s = time.monotonic() - started
                phase_started = time.monotonic()
                total = sum(report["supervised_tokens"] for report in reports.values())
                reduce_gradients([buffers[f"grad-{i}"] for i in range(len(workers))], buffers["aggregate"], total)
                reduce_s = time.monotonic() - phase_started
                phase_started = time.monotonic()
                channels[0].send(command="update", step=step)
                update = channels[0].receive(timeout)
                if update["event"] != "updated":
                    raise RuntimeError(f"unexpected update response: {update}")
                update_s = time.monotonic() - phase_started
                phase_started = time.monotonic()
                for channel in channels:
                    channel.send(command="sync", initialize_optimizer=False)
                for channel in channels:
                    assert channel.receive(timeout)["event"] == "synced"
                sync_s = time.monotonic() - phase_started
                anchor_micro_s = reports[anchor]["compute_s"] / reports[anchor]["microbatches"]
                # Include export/tail waiting in the overhead estimate, not just Adam/Muon.
                overhead_s = max(0.0, accumulation_s - reports[anchor]["compute_s"]) + reduce_s + update_s + sync_s
                anchor_microbatch_ema = (anchor_micro_s if anchor_microbatch_ema is None
                                        else 0.8 * anchor_microbatch_ema + 0.2 * anchor_micro_s)
                barrier_ema = overhead_s if barrier_ema is None else 0.8 * barrier_ema + 0.2 * overhead_s
                target = next_anchor_target(base_target, maximum_target, anchor_microbatch_ema,
                                            barrier_ema, config.get("target_compute_fraction"))
                elapsed = time.monotonic() - started
                worker_metrics = {}
                for index, worker in enumerate(workers):
                    report = reports[index]
                    worker_metrics[worker["name"]] = {
                        key: report[key] for key in ("microbatches", "supervised_tokens", "compute_s", "export_s", "wait_for_stop_s")
                    }
                    worker_metrics[worker["name"]]["compute_fraction"] = min(1.0, report["compute_s"] / elapsed)
                    worker_metrics[worker["name"]]["supervised_tokens_per_compute_s"] = report["supervised_tokens"] / max(report["compute_s"], 1e-12)
                result = {"step": step, "loss": sum(r["loss_sum"] for r in reports.values()) / total,
                          "supervised_tokens": total, "grad_norm": update["grad_norm"],
                          "microbatches": {w["name"]: reports[i]["microbatches"] for i, w in enumerate(workers)},
                          "time_s": elapsed, "anchor_target": round_target, "next_anchor_target": target,
                          "workers": worker_metrics,
                          "timing_s": {"accumulation": accumulation_s, "reduce": reduce_s,
                                       "update": update_s, "sync": sync_s}}
                if step == 1:
                    result["warmup"] = warmup
                results.append(result)
                print(json.dumps(result, allow_nan=False), flush=True)
            if config.get("checkpoint_dir"):
                channels[0].send(command="save", results=results)
                assert channels[0].receive(timeout)["event"] == "saved"
            for channel in channels:
                channel.send(command="stop")
            for process in processes:
                process.wait(timeout=10)
                if process.returncode:
                    raise RuntimeError(f"worker exited with status {process.returncode}")
        finally:
            for channel in channels:
                channel.sock.close()
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            for log in logs:
                log.close()
    return results


def worker_main(config, index, fd):
    # Backend environment is fixed by the launcher BEFORE any mini_llm import.
    from dataclasses import asdict, replace
    from mini_llm.backend import xp, BACKEND_NAME, asnumpy, synchronize, validate_bfloat16_backend
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.model_registry import registry
    from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata
    from mini_llm.optim.hybrid_muon import HybridMuonAdamW
    from mini_llm.optim.grad_clip import clip_grad_global_norm

    channel = Channel(socket.socket(fileno=fd))
    worker = config["workers"][index]
    if BACKEND_NAME == "cupy":
        xp.cuda.Device(int(worker.get("device", 0))).use()
        runtime = "hip" if xp.cuda.runtime.is_hip else "cuda"
        if worker.get("expected_runtime", runtime) != runtime:
            raise RuntimeError(f"expected {worker['expected_runtime']}, got {runtime}")
    dtype = config.get("dtype", "bfloat16")
    if dtype not in {"bfloat16", "float32"}:
        raise ValueError("only BF16 mixed or FP32 training is supported")
    if dtype == "bfloat16":
        validate_bfloat16_backend()
    model_config = replace(registry.config("model_b_348m_sparse16"), dtype=dtype)
    if config.get("smoke"):
        # Keep the production attention ABI: Triton specializes 16 Q / 4 KV
        # heads of width 64, local window 512, and aligned pooling blocks.
        model_config = replace(model_config, tokenizer_vocab_size=64, context_length=512,
            n_layers=2, d_ff=128,
            dual_resolution=replace(model_config.dual_resolution,
                sparse_retrieval_layers=(0, 1)))
    if worker["context_length"] > model_config.context_length:
        raise ValueError("worker context exceeds model configuration")
    model = DecoderLanguageModel(model_config, rng_seed=int(config.get("seed", 42)), dtype=dtype)
    parameters = list(model.parameters())
    if config.get("init_from") and index == 0:
        from mini_llm.model_loading import read_native_config
        from mini_llm.checkpoint import load_checkpoint
        if read_native_config(Path(config["init_from"])) != model_config:
            raise ValueError("initial checkpoint architecture/dtype must exactly match")
        saved, _, _ = load_checkpoint(config["init_from"], skip_optimizer=True)
        if set(saved) != {p.name for p in parameters}:
            raise ValueError("initial checkpoint parameter names disagree")
        for parameter in parameters:
            value = saved.pop(parameter.name)
            if value.shape != parameter.data.shape:
                raise ValueError(f"initial checkpoint shape mismatch: {parameter.name}")
            parameter.data[...] = value
    channel.send(event="ready", schema=[(p.name, list(p.shape)) for p in parameters])
    rng = np.random.default_rng(int(config.get("seed", 42)) + 1009 * (index + 1))
    shards, document_indices = [], []
    eos = int(config.get("eos_token_id", model_config.vocab_size - 1))
    if not 0 <= eos < model_config.vocab_size:
        raise ValueError("eos_token_id outside vocabulary")
    if not config.get("smoke"):
        from mini_llm.data.token_shards import map_token_shard, build_packed_document_index
        for path in config["train_shards"]:
            shard = map_token_shard(str(ROOT / path))
            shards.append(shard)
            document_indices.append(build_packed_document_index(shard, eos))
    optimizer = None
    buffers = {}
    loss_sum = supervised = 0.0
    microbatches = 0
    compute_s = 0.0

    def copy_buffer(name, attribute, outward):
        offset = 0
        for parameter in parameters:
            array = getattr(parameter, attribute)
            size = int(array.size)
            host = buffers[name][offset:offset + size].reshape(array.shape)
            if outward:
                host[...] = asnumpy(array.astype("float32", copy=False))
            else:
                array[...] = xp.asarray(host, dtype=array.dtype)
            offset += size
        synchronize()

    def microbatch():
        nonlocal loss_sum, supervised, microbatches, compute_s
        micro_started = time.monotonic()
        if config.get("smoke"): 
            raw = rng.integers(0, model_config.vocab_size - 1,
                size=(worker["batch_size"], worker["context_length"] + 1), dtype=np.int64)
            inputs, targets = raw[:, :-1].copy(), raw[:, 1:].copy()
            mask = np.ones(inputs.shape, dtype=np.float32)
        else:
            from mini_llm.data.token_shards import create_dual_resolution_document_minibatch
            slot = int(rng.integers(len(shards)))
            inputs, targets, metadata = create_dual_resolution_document_minibatch(
                shards[slot], worker["batch_size"], worker["context_length"], rng,
                eos_token_id=eos, document_index=document_indices[slot], return_metadata=True,
                shape_bucket=worker.get("shape_bucket", "fine"))
            mask = metadata["target_loss_mask"]
        count = float(mask.sum())
        metadata = build_dual_resolution_metadata(inputs, eos, model_config.dual_resolution.block_size)
        head, cache = model.forward_body(xp.asarray(inputs),
            activation_checkpoint=bool(worker.get("activation_checkpoint", True)),
            dual_resolution_metadata=metadata)
        loss, loss_cache = model.chunked_lm_head_loss_forward(head, xp.asarray(targets),
            chunk_tokens=int(worker.get("lm_head_chunk_tokens", 512)),
            loss_mask=xp.asarray(mask), return_device_loss=True)
        value = float(loss.item())
        if not math.isfinite(value):
            raise ValueError("nonfinite microbatch loss; update aborted")
        # Native CE is a token MEAN. Undo that normalisation for local SUM grads.
        dx = model.chunked_lm_head_backward(loss_cache, grad_scale=count)
        model.backward_body(dx, cache)
        synchronize()
        loss_sum += value * count
        supervised += count
        microbatches += 1
        compute_s += time.monotonic() - micro_started

    while True:
        message = channel.receive(float(config.get("timeout_s", 1800)))
        command = message["command"]
        if command == "stop":
            return
        if command == "map":
            directory = Path(message["directory"])
            for name in ("weights", "aggregate", f"grad-{index}"):
                buffers[name] = np.memmap(directory / name, mode="r+", dtype=np.float32, shape=(message["count"],))
            channel.send(event="mapped")
        elif command == "publish":
            copy_buffer("weights", "data", True)
            channel.send(event="published")
        elif command == "sync":
            # The owner already has the updated weights and refreshed buffers.
            if index != 0 or message["initialize_optimizer"]:
                copy_buffer("weights", "data", False)
                model.refresh_compute_buffers()
            model.zero_grad()
            if message["initialize_optimizer"] and index == 0:
                options = dict(lr=1.5e-4, beta1=0.9, beta2=0.95, eps=1e-8,
                    weight_decay=0.1, muon_matmul_precision="bf16" if dtype == "bfloat16" else "fp32")
                options.update(config.get("optimizer", {}))
                optimizer = HybridMuonAdamW(list(model.optimization_parameters()), **options)
            channel.send(event="synced")
        elif command == "warmup":
            sampling_state = copy.deepcopy(rng.bit_generator.state)
            loss_sum = supervised = compute_s = 0.0
            microbatches = 0
            for _ in range(message["microbatches"]):
                model.zero_grad()
                microbatch()
            elapsed_compute = compute_s
            rng.bit_generator.state = sampling_state
            model.zero_grad()
            loss_sum = supervised = compute_s = 0.0
            microbatches = 0
            channel.send(event="warmed", microbatches=message["microbatches"], compute_s=elapsed_compute)
        elif command == "begin":
            model.zero_grad()
            loss_sum = supervised = compute_s = 0.0
            microbatches = 0
            target = message["target_microbatches"]
            wait_for_stop_s = 0.0
            if target is not None:
                for _ in range(target):
                    microbatch()
                # Notify before export so peers can stop while this copy runs.
                channel.send(event="accumulated")
            else:
                cap = worker.get("max_microbatches")
                while True:
                    microbatch()
                    wait_started = time.monotonic()
                    capped = cap is not None and microbatches >= cap
                    if capped or b"\n" in channel.buffer or select.select([channel.sock], [], [], 0)[0]:
                        stop = channel.receive(float(config.get("timeout_s", 1800)))
                        wait_for_stop_s += time.monotonic() - wait_started
                        if stop["command"] != "finish":
                            raise RuntimeError(f"unexpected accumulation command: {stop}")
                        break
            export_started = time.monotonic()
            copy_buffer(f"grad-{index}", "grad", True)
            channel.send(event="exported", loss_sum=loss_sum, supervised_tokens=supervised,
                         microbatches=microbatches, compute_s=compute_s,
                         export_s=time.monotonic() - export_started, wait_for_stop_s=wait_for_stop_s)
        elif command == "update" and index == 0:
            copy_buffer("aggregate", "grad", False)
            norm, _, finite = clip_grad_global_norm(list(model.optimization_parameters()),
                max_norm=float(config.get("grad_clip", 1.0)))
            if not finite:
                raise ValueError("nonfinite aggregate norm; update aborted")
            optimizer.step()
            model.refresh_compute_buffers()
            copy_buffer("weights", "data", True)
            channel.send(event="updated", grad_norm=float(norm.item()))
        elif command == "save" and index == 0:
            from mini_llm.checkpoint import save_checkpoint
            path = Path(config["checkpoint_dir"])
            if path.exists():
                raise FileExistsError("use a NEW checkpoint_dir; never overwrite a production checkpoint")
            path.mkdir(parents=True)
            (path / "config.json").write_text(json.dumps(asdict(model_config), indent=2))
            save_checkpoint(path, {p.name: p.data for p in parameters}, optimizer.checkpoint_state(),
                {"step": optimizer.step_index, "heterogeneous_config": config,
                 "heterogeneous_results": message["results"], "backend": "cupy",
                 "gradient_accumulation_weighting": "supervised_tokens",
                 "exact_resume_supported": False})
            channel.send(event="saved")
        else:
            raise RuntimeError(f"unexpected worker command: {command}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--work-dir", help="existing host scratch directory (mmap files removed on exit)")
    parser.add_argument("--worker-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.worker_fd is not None:
        worker_main(config, args.worker_index, args.worker_fd)
    else:
        run(config, steps=args.steps, work_dir=args.work_dir)


if __name__ == "__main__":
    main()
