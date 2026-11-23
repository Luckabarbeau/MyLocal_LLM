"""Optional native accumulation peer for the established production trainer.

The owner alone restores/updates/saves optimizer state. The peer uses the same
ExtendedTrainer sampler and handwritten backward without allocating an optimizer.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from contextlib import nullcontext
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
import traceback
import uuid

import numpy as np

from train_heterogeneous import Channel, ROOT


# A rejected peer export is replayed on IDENTICAL data so the recurrence can be
# attributed: reproducing on the same documents and the same peer membership
# points at a data/code-dependent defect, while a clean replay points at non-deterministic
# execution on the peer GPU. The budget bounds the cost (one extra round per rejection)
# and keeps a persistently faulty card from replaying one round forever; once it is
# exhausted the established fresh-sample retry resumes.
PEER_REPLAY_BUDGET = 2

SAMPLER_FIELDS = (
    "current_train_shard_idx", "current_train_source_shard_idx",
    "train_source_batch_counts", "train_source_supervised_tokens", "tokens_processed",
    "sample_trace_sha256", "sample_trace_microbatches",
)


def sampler_state(trainer):
    state = {name: copy.deepcopy(getattr(trainer, name)) for name in SAMPLER_FIELDS}
    state["train_rng_state"] = copy.deepcopy(trainer.train_rng.bit_generator.state)
    return state


def restore_sampler(trainer, state):
    for name in SAMPLER_FIELDS:
        if name in state:
            setattr(trainer, name, copy.deepcopy(state[name]))
    trainer.train_rng.bit_generator.state = copy.deepcopy(state["train_rng_state"])


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _copy(parameters, buffer, attribute, outward):
    from .backend import xp, asnumpy, synchronize
    offset = 0
    for parameter in parameters:
        array = getattr(parameter, attribute)
        count = int(array.size)
        host = buffer[offset:offset + count].reshape(array.shape)
        if outward:
            host[...] = asnumpy(array.astype("float32", copy=False))
        else:
            array[...] = xp.asarray(host, dtype=array.dtype)
        offset += count
    synchronize()


def _sources(observations):
    from .diagnostics.source_metrics import empty_record, merge_record
    from .backend import asnumpy
    result = {}
    for name, coverage, loss in observations:
        record = dict(coverage)
        record["loss_nll_sum"] = float(asnumpy(loss)) * float(record["supervised_token_weight"])
        merge_record(result.setdefault(name, empty_record()), record)
    return result


def _start_diagnostics(trainer, settings, role):
    if not settings["enabled"]:
        return None
    from .diagnostics.gpu_events import GpuEventLog
    log = GpuEventLog(trainer, role=role, directory=settings["directory"], run_id=settings["run_id"])
    try:
        log.install_model_hooks(trainer.model)
        trainer.gpu_diagnostic_log = log
        log.emit("session_begin", trainer_step=trainer.step,
                 optimizer_step=getattr(trainer.optimizer, "step_index", None),
                 sampler_state=sampler_state(trainer))
    except BaseException:
        log.close()
        trainer.gpu_diagnostic_log = None
        raise
    return log


def _diagnostic_role(prefix):
    from .backend import BACKEND_NAME, xp
    runtime = "numpy" if BACKEND_NAME == "numpy" else "hip" if xp.cuda.runtime.is_hip else "cuda"
    return f"{prefix}-{runtime}"


class _Warmup:
    def __init__(self, count):
        self.count = count

    def begin_step(self, trainer):
        pass

    def microsteps(self, trainer):
        return range(self.count)

    def synchronize_gradients(self, trainer, **stats):
        return stats


def warmup(trainer, count):
    """Compile sampled shapes without advancing sampling, optimizer, or monitors."""
    if not count:
        return
    state = sampler_state(trainer)
    enabled = trainer.optimizer_enabled
    previous_hook = getattr(trainer, "accumulation_peer", None)
    monitors = {name: getattr(trainer, name) for name in (
        "gradient_noise", "gradient_convergence", "component_updates")}
    diagnostics = getattr(trainer, "gpu_diagnostic_log", None)
    old_phase = diagnostics.phase if diagnostics else None
    if diagnostics is not None:
        diagnostics.phase = "warmup"
        diagnostics.emit("warmup_begin", microbatches=count)
    try:
        trainer.optimizer_enabled = False
        trainer.accumulation_peer = _Warmup(count)
        for name in monitors:
            setattr(trainer, name, None)
        trainer.model.zero_grad()
        trainer.train_step()
    finally:
        restore_sampler(trainer, state)
        trainer.optimizer_enabled = enabled
        trainer.accumulation_peer = previous_hook
        for name, monitor in monitors.items():
            setattr(trainer, name, monitor)
        trainer.model.zero_grad()
        if diagnostics is not None:
            diagnostics.emit("warmup_end", microbatches=count, sampler_restored=True)
            diagnostics.phase = old_phase


class ProductionPeerSession:
    """One owner/anchor plus one independently accumulating native peer."""
    def __init__(self, trainer, *, config_path, training_state, acknowledge_fork, tokenizer_path):
        config = json.loads(Path(config_path).read_text())
        if not acknowledge_fork and not training_state.get("heterogeneous_state"):
            raise ValueError("adding a peer changes the training batch; pass --acknowledge-heterogeneous-fork")
        if trainer.training_objective is not None or trainer.gradient_accumulation_weighting != "supervised_tokens":
            raise ValueError("production peer requires AR supervised-token accumulation")
        if (trainer.ar_accum_length_buckets or trainer.dual_accum_length_buckets
                or trainer.streaming_cache is not None or trainer.progressive_growth_steps
                or trainer.full_history_only_until_step is not None
                or trainer.gradient_noise is not None or trainer.gradient_convergence is not None):
            raise ValueError("production peer does not support bucketing, remote streams, depth/freezing schedules, or local gradient-noise monitors")
        if not trainer.model.config.dual_resolution.sparse_retrieval_stride:
            raise ValueError("this production integration requires sparse Model B")
        self.config = config
        self.timeout = float(config.get("timeout_s", 1800))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("peer timeout_s must be finite and positive")
        for key in ("batch_size", "context_length", "max_microbatches"):
            value = config.get(key, trainer.batch_size if key == "batch_size" else trainer.seq_length if key == "context_length" else None)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"peer {key} must be a positive integer")
        if int(config.get("context_length", trainer.seq_length)) > trainer.model.config.context_length:
            raise ValueError("peer context exceeds checkpoint model capacity")
        warmup_count = config.get("warmup_microbatches", 2)
        if type(warmup_count) is not int or warmup_count < 0:
            raise ValueError("peer warmup_microbatches must be a nonnegative integer")
        self.contract = {
            "version": 1, "peer": config, "owner_batch_size": trainer.batch_size,
            "owner_context_length": trainer.seq_length, "owner_microbatches": trainer.grad_accum_steps,
            "mixture_sha256": trainer.local_mixture_manifest_sha256,
            "normalization": "global_supervised_tokens", "owner_is_anchor": True,
        }
        saved = training_state.get("heterogeneous_state")
        if saved and saved["contract"] != self.contract:
            print("WARNING: heterogeneous topology/data contract changed; "
                  "continuing with current settings", flush=True)
        self.parent_step = int(saved.get("parent_step", trainer.step) if saved else trainer.step)
        self.history = copy.deepcopy(saved.get("history", []) if saved else [])
        if not saved:
            self.history.append({"step": trainer.step, "change": "single_device_to_native_accumulation_peer",
                                 "optimizer_state_preserved": True, "bitwise_trajectory_preserved": False})
        self.process = None
        self.channel = None
        self.log = None
        self.diagnostics = None
        self.round_sampler = None
        self.pending_replay = None
        self.replay_budget = PEER_REPLAY_BUDGET
        self._trainer = trainer
        enabled = os.environ.get("NATIVE_GPU_LOG", "1")
        if enabled not in {"0", "1"}:
            raise ValueError("NATIVE_GPU_LOG must be 0 or 1")
        diagnostic_settings = {
            "enabled": enabled == "1", "run_id": uuid.uuid4().hex,
            "directory": str(Path(os.environ.get("NATIVE_GPU_LOG_DIR",
                str(Path(config.get("log_file", "logs/model_b_heterogeneous_peer.log")).parent / "gpu_diagnostics"))).resolve()),
        }
        self.directory = tempfile.TemporaryDirectory(prefix="production-peer-", dir=config.get("work_dir"))
        directory = Path(self.directory.name)
        self.parameters = list(trainer.model.parameters())
        schema = [(p.name, list(p.shape)) for p in self.parameters]
        count = sum(int(p.size) for p in self.parameters)
        self.buffers = {name: np.memmap(directory / name, mode="w+", dtype=np.float32, shape=(count,))
                        for name in ("weights", "peer-gradient")}
        try:
            self.diagnostics = _start_diagnostics(trainer, diagnostic_settings, _diagnostic_role("owner"))
            if self.diagnostics is not None:
                print(f"Per-GPU diagnostic log: {self.diagnostics.path}", flush=True)
            kwargs = dict(
                train_shard_paths=trainer.train_shard_paths, val_shard_paths=trainer.val_shard_paths,
                train_source_shards=trainer.train_source_shards, val_source_shards=trainer.val_source_shards,
                source_weights=trainer.source_weights,
                batch_size=config.get("batch_size", trainer.batch_size),
                seq_length=config.get("context_length", trainer.seq_length), grad_accum_steps=1,
                gradient_accumulation_weighting="supervised_tokens", eos_token_id=trainer.eos_token_id,
                rng_seed=config.get("seed", 104729), optimizer_enabled=False,
                total_steps=trainer.total_steps, optimizer_kind=trainer.optimizer.kind,
                local_mixture_manifest_sha256=trainer.local_mixture_manifest_sha256,
            )
            setup = dict(model_config=asdict(trainer.model.config), trainer_kwargs=_json_value(kwargs),
                         schema=schema, count=count, directory=str(directory), config=config,
                         sampler_state=saved.get("peer_sampler") if saved else None,
                         step=trainer.step, diagnostics=diagnostic_settings)
            setup_path = directory / "setup.json"
            setup_path.write_text(json.dumps(setup))
            env = os.environ.copy()
            for key in list(env):
                if key.startswith("MINI_LLM_"):
                    del env[key]
            # Start from the owner's restored numerical profile, then explicit HIP overrides.
            env.update({key: value for key, value in os.environ.items() if key.startswith("MINI_LLM_")})
            env.update(config.get("env", {}))
            env["MINI_LLM_BACKEND"] = config.get("backend", "cupy")
            if env["MINI_LLM_BACKEND"] not in {"cupy", "numpy"}:
                raise ValueError("peer requires native CuPy (or NumPy tests), never PyTorch")
            log_path = Path(config.get("log_file", "logs/model_b_heterogeneous_peer.log"))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log = log_path.open("a")
            parent, child = socket.socketpair()
            try:
                self.process = subprocess.Popen(
                    [str(ROOT / config["python"]), "-u", "-m", "mini_llm.heterogeneous_production",
                     "--fd", str(child.fileno()), "--setup", str(setup_path)],
                    cwd=ROOT, env=env, pass_fds=(child.fileno(),), stdout=self.log, stderr=subprocess.STDOUT)
            except BaseException:
                parent.close()
                raise
            finally:
                child.close()
            self.channel = Channel(parent)
            ready = self._expect("ready")
            if ready.get("event") != "ready" or ready["schema"] != _json_value(schema):
                raise ValueError("peer parameter schema mismatch")
            _copy(self.parameters, self.buffers["weights"], "data", True)
            self.channel.send(command="initialize")
            self._expect("synced")
            self.channel.send(command="warmup", count=warmup_count)
            warmup(trainer, warmup_count)
            self._expect("warmed")
            if self.diagnostics is not None:
                self.diagnostics.emit("peer_ready", peer_pid=self.process.pid,
                                      contract=self.contract, peer_stdout_log=str(log_path))
            print(f"Native production peer ready: parent step={trainer.step}, context={kwargs['seq_length']}, "
                  f"batch={kwargs['batch_size']}; optimizer state remains owner-only", flush=True)
        except BaseException:
            self.close()
            raise

    def _expect(self, event):
        try:
            message = self.channel.receive(self.timeout)
            if message.get("event") != event:
                raise RuntimeError(f"expected peer {event}, got {message}")
            return message
        except Exception as exc:
            if self.diagnostics is not None:
                self.diagnostics.emit("peer_transport_error", expected_event=event,
                    exception_type=type(exc).__name__, message=str(exc),
                    traceback=traceback.format_exc(), peer_returncode=self.process.poll())
            raise

    def begin_step(self, trainer):
        self.started = time.monotonic()
        self.round_sampler = sampler_state(trainer)
        self.channel.send(command="begin", step=trainer.step)
        if self.diagnostics is not None:
            self.diagnostics.emit("peer_begin_sent", trainer_step=trainer.step)

    def microsteps(self, trainer):
        for index in range(trainer.grad_accum_steps):
            yield index
            if (index + 1) % 64 == 0:
                print(f"Owner accumulation step={trainer.step + 1} microbatch={index + 1}/{trainer.grad_accum_steps}", flush=True)
            if self.process.poll() is not None:
                raise RuntimeError("production peer exited; inspect peer log; owner update aborted")

    def synchronize_gradients(self, trainer, *, avg_loss, supervised_tokens, executed_tokens,
                              microbatches, source_observations):
        from .backend import xp
        if self.diagnostics is not None:
            self.diagnostics.emit("gradient_exchange_begin", local_loss=avg_loss,
                                  local_microbatches=microbatches, local_supervised_tokens=supervised_tokens)
        self.channel.send(command="finish")
        report = self._expect("exported")
        if report["step"] != trainer.step:
            raise ValueError("peer gradient weight version mismatch")
        peer_tokens = float(report["supervised_tokens"])
        total = supervised_tokens + peer_tokens
        if not math.isfinite(total) or peer_tokens <= 0 or supervised_tokens <= 0:
            raise ValueError("invalid global token count")
        if not math.isfinite(report["avg_loss"]):
            raise ValueError("nonfinite peer loss; update aborted")
        # Both trainers expose locally token-normalized, UNCLIPPED gradients here.
        # Weight means by their true supervised-token counts, never by rank count.
        offset = 0
        reduction_started = time.monotonic()
        bad_ranges = []
        for index, parameter in enumerate(self.parameters):
            size = int(parameter.size)
            flat = parameter.grad.reshape(-1)
            flat *= supervised_tokens / total
            nan_count = inf_count = 0
            for start in range(0, size, 1048576):
                stop = min(size, start + 1048576)
                host = self.buffers["peer-gradient"][offset + start:offset + stop]
                if not np.isfinite(host).all():
                    nan_count += int(np.isnan(host).sum())
                    inf_count += int(np.isinf(host).sum())
                flat[start:stop] += xp.asarray(host) * (peer_tokens / total)
            if nan_count or inf_count:
                bad_ranges.append((parameter.name or f"parameter_{index}", offset, size,
                                   nan_count, inf_count))
            offset += size
        self.last_report = report
        self.last_report["reduce_s"] = time.monotonic() - reduction_started
        if bad_ranges:
            return self._reject_peer_export(trainer, report, bad_ranges)
        self._peer_replay_verdict(report, [])
        for name, coverage in report["source_metrics"].items():
            weight = float(coverage["supervised_token_weight"])
            source_observations.append((name, coverage, xp.asarray(coverage["loss_nll_sum"] / weight)))
            trainer.train_source_supervised_tokens[name] = trainer.train_source_supervised_tokens.get(name, 0) + coverage["supervised_tokens"]
            if name in trainer.train_source_batch_counts:
                trainer.train_source_batch_counts[name] += coverage["microbatches"]
        trainer.tokens_processed += report["executed_tokens"]
        if self.diagnostics is not None:
            self.diagnostics.emit("gradient_exchange_complete", owner_microbatches=microbatches,
                peer_microbatches=report["microbatches"], owner_supervised_tokens=supervised_tokens,
                peer_supervised_tokens=peer_tokens, global_supervised_tokens=total,
                peer_loss=report["avg_loss"], reduce_s=self.last_report["reduce_s"])
        return dict(loss=(avg_loss * supervised_tokens + report["avg_loss"] * peer_tokens) / total,
                    supervised_tokens=total, executed_tokens=executed_tokens + report["executed_tokens"],
                    nominal_tokens=trainer.batch_size * trainer.target_seq_length * microbatches + report["nominal_tokens"])

    def _reject_peer_export(self, trainer, report, bad_ranges):
        """Discard one peer accumulation round whose export was non-finite.

        This reuses the established owner-side treatment of a non-finite gradient:
        `train_step` clears both gradients, releases the waiting peer with
        ``updated=False``, and retries the same step from fresh samples, so the
        weights, optimizer and scheduler state are untouched. The always-on
        shared-buffer scan already localizes the offending parameter, which the
        owner's combined-gradient audit cannot recover once the Inf is folded in.
        Only rejected rounds pay for the extra magnitude scan.
        """
        buffer = self.buffers["peer-gradient"]
        rows = []
        for name, start, size, nan_count, inf_count in bad_ranges:
            max_finite_abs = 0.0
            for offset in range(start, start + size, 1048576):
                stop = min(start + size, offset + 1048576)
                host = buffer[offset:stop]
                finite = host[np.isfinite(host)]
                if finite.size:
                    max_finite_abs = max(max_finite_abs, float(np.abs(finite).max()))
            rows.append({"name": name, "nonfinite": nan_count + inf_count, "nan": nan_count,
                         "inf": inf_count, "max_finite_abs": max_finite_abs})
        print(f"Peer gradient audit: parameters={len(rows)}, elements={sum(int(p.size) for p in self.parameters):,}", flush=True)
        for row in rows:
            print(
                "  NONFINITE "
                f"{row['name']}: count={row['nonfinite']:,}, nan={row['nan']:,}, "
                f"inf={row['inf']:,}, max_finite_abs={row['max_finite_abs']:.6e}"
            )
        if self.diagnostics is not None:
            self.diagnostics.emit("gradient_rejected", completed_step=trainer.step,
                attempted_step=trainer.step + 1, failure_kind="peer_nonfinite_export",
                peer_loss=report["avg_loss"], peer_microbatches=report["microbatches"],
                peer_supervised_tokens=report["supervised_tokens"],
                peer_gradient_audit={"all_finite": False, "nonfinite": rows})
        self._peer_replay_verdict(report, [row["name"] for row in rows])
        replay_queued = self._queue_identical_replay(trainer, report, rows)
        return dict(rejected=dict(reason="peer_nonfinite_export", parameters=rows,
                                   replay_queued=replay_queued))

    def _peer_replay_verdict(self, report, names):
        """Classify the previous rejection: did the identical replay fail again?

        Membership identity matters: the peer's microbatch count is normally timing
        dependent, so a clean replay that also saw a different number of
        microbatches is inconclusive rather than evidence of instability.
        """
        prior = self.pending_replay
        if prior is None:
            return None
        self.pending_replay = None
        membership_match = (int(report["microbatches"]) == prior["peer_microbatches"]
                            and float(report["supervised_tokens"]) == prior["peer_supervised_tokens"])
        if names:
            reproduced = sorted(names) == sorted(prior["parameters"])
            verdict = "reproduced_same_parameters" if reproduced else "reproduced_different_parameters"
        else:
            verdict = "clean_replay" if membership_match else "clean_replay_membership_differs"
        record = dict(
            event="peer_replay_verdict", verdict=verdict, membership_match=membership_match,
            prior_parameters=prior["parameters"], replay_parameters=names,
            prior_peer_microbatches=prior["peer_microbatches"],
            prior_peer_supervised_tokens=prior["peer_supervised_tokens"],
            replay_peer_microbatches=int(report["microbatches"]),
            replay_peer_supervised_tokens=float(report["supervised_tokens"]),
            replay_budget_remaining=self.replay_budget,
        )
        print(
            f"PEER REPLAY VERDICT: {verdict} "
            f"(peer microbatches {prior['peer_microbatches']} -> {record['replay_peer_microbatches']}, "
            f"membership_match={membership_match}, remaining_budget={self.replay_budget})"
        )
        if verdict == "clean_replay":
            print("PEER REPLAY VERDICT: the same data on the same peer GPU was clean, so the "
                  "non-finite export is NOT reproducible; suspect non-deterministic GPU execution",
                  flush=True)
        if self.diagnostics is not None:
            self.diagnostics.emit(**record)
        return record

    def _queue_identical_replay(self, trainer, report, rows):
        """Roll both samplers back so the retry re-runs this exact round."""
        if self.replay_budget <= 0 or self.round_sampler is None:
            if self.diagnostics is not None:
                self.diagnostics.emit("peer_replay_exhausted", trainer_step=trainer.step,
                                      replay_budget=PEER_REPLAY_BUDGET)
            print(f"PEER REPLAY: budget exhausted; resuming the established fresh-sample retry",
                  flush=True)
            return False
        self.replay_budget -= 1
        restore_sampler(trainer, self.round_sampler)
        self.pending_replay = dict(
            parameters=[row["name"] for row in rows],
            peer_microbatches=int(report["microbatches"]),
            peer_supervised_tokens=float(report["supervised_tokens"]),
        )
        if self.diagnostics is not None:
            self.diagnostics.emit("peer_replay_queued", trainer_step=trainer.step,
                                  parameters=self.pending_replay["parameters"],
                                  owner_microbatches=trainer.grad_accum_steps,
                                  peer_microbatches=self.pending_replay["peer_microbatches"],
                                  peer_supervised_tokens=self.pending_replay["peer_supervised_tokens"],
                                  replay_budget_remaining=self.replay_budget)
        print(
            f"PEER REPLAY: queued identical-data retry "
            f"(owner microbatches={trainer.grad_accum_steps}, "
            f"peer microbatches={self.pending_replay['peer_microbatches']}, "
            f"remaining_budget={self.replay_budget})", flush=True)
        return True

    def finish_step(self, trainer, *, updated):
        if updated:
            with self.diagnostics.stage("weight_publish") if self.diagnostics else nullcontext():
                _copy(self.parameters, self.buffers["weights"], "data", True)
        replay = self.pending_replay if not updated else None
        self.channel.send(command="sync", updated=updated, step=trainer.step + int(updated),
                          restore_sampler=replay is not None,
                          replay_microbatches=(replay or {}).get("peer_microbatches"))
        self._expect("synced")
        if self.diagnostics is not None:
            self.diagnostics.emit("weights_synced", updated=updated,
                                  new_weight_version=trainer.step + int(updated),
                                  peer_microbatches=self.last_report["microbatches"])
        print(f"HETEROGENEOUS step={trainer.step + int(updated)} owner_microbatches={trainer.grad_accum_steps} "
              f"peer_microbatches={self.last_report['microbatches']} peer_tokens={self.last_report['supervised_tokens']:.0f} "
              f"updated={updated} elapsed_s={time.monotonic() - self.started:.3f}", flush=True)

    def state_dict(self, trainer):
        self.channel.send(command="state")
        state = self._expect("state")
        if self.diagnostics is not None:
            self.diagnostics.emit("checkpoint_peer_state_captured", trainer_step=trainer.step,
                                  peer_sampler=state["sampler_state"])
        return dict(contract=self.contract, parent_step=self.parent_step, history=self.history,
                    peer_sampler=state["sampler_state"], step=trainer.step,
                    trajectory_note="optimizer/scheduler continued; timing-dependent peer membership changes trajectory")

    def close(self):
        if self.process is not None:
            if self.process.poll() is None and self.channel is not None:
                try:
                    self.channel.send(command="stop")
                    self.process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    self.process.terminate()
            if self.process.poll() is None:
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        if self.channel is not None:
            self.channel.sock.close()
        if self.log is not None:
            self.log.close()
        self.buffers.clear()
        self.directory.cleanup()
        if self.diagnostics is not None:
            try:
                self.diagnostics.emit("session_end", trainer_step=self._trainer.step,
                                      peer_returncode=self.process.poll() if self.process else None)
            finally:
                self.diagnostics.close()
                self._trainer.gpu_diagnostic_log = None
                self.diagnostics = None


class _PeerHooks:
    def __init__(self, channel, setup, buffers):
        self.channel, self.setup, self.buffers = channel, setup, buffers
        self.timeout = float(setup["config"].get("timeout_s", 1800))
        self.round_sampler = None
        self.replay_cap = None

    def begin_step(self, trainer):
        self.started = time.monotonic()
        self.round_sampler = sampler_state(trainer)

    def microsteps(self, trainer):
        index = 0
        cap = self.replay_cap if self.replay_cap is not None else self.setup["config"].get("max_microbatches")
        self.replay_cap = None
        while True:
            yield index
            index += 1
            if index % 64 == 0:
                print(f"Peer accumulation step={trainer.step + 1} microbatch={index}", flush=True)
            if ((cap is not None and index >= cap) or b"\n" in self.channel.buffer
                    or select.select([self.channel.sock], [], [], 0)[0]):
                message = self.channel.receive(self.timeout)
                if message["command"] != "finish":
                    raise RuntimeError(f"unexpected peer stop: {message}")
                diagnostics = getattr(trainer, "gpu_diagnostic_log", None)
                if diagnostics is not None:
                    diagnostics.emit("accumulation_stopped", completed_microbatches=index,
                                     cap_reached=cap is not None and index >= cap)
                return

    def synchronize_gradients(self, trainer, *, avg_loss, supervised_tokens, executed_tokens,
                              microbatches, source_observations):
        parameters = list(trainer.model.parameters())
        diagnostics = getattr(trainer, "gpu_diagnostic_log", None)
        with diagnostics.stage("gradient_export") if diagnostics else nullcontext():
            _copy(parameters, self.buffers["peer-gradient"], "grad", True)
        if diagnostics is not None:
            diagnostics.emit("gradient_export_complete", local_loss=avg_loss,
                local_supervised_tokens=supervised_tokens, local_executed_tokens=executed_tokens,
                local_microbatches=microbatches)
        self.channel.send(event="exported", step=trainer.step, avg_loss=avg_loss,
                          supervised_tokens=supervised_tokens, executed_tokens=executed_tokens,
                          microbatches=microbatches,
                          nominal_tokens=trainer.batch_size * trainer.target_seq_length * microbatches,
                          source_metrics=_sources(source_observations), compute_export_s=time.monotonic() - self.started)
        with diagnostics.stage("await_owner_update") if diagnostics else nullcontext():
            message = self.channel.receive(self.timeout)
        if message["command"] != "sync":
            raise RuntimeError(f"unexpected peer sync: {message}")
        if message["updated"]:
            with diagnostics.stage("weight_import") if diagnostics else nullcontext():
                _copy(parameters, self.buffers["weights"], "data", False)
                trainer.model.refresh_compute_buffers()
        trainer.model.zero_grad()
        trainer.step = int(message["step"])
        if message.get("restore_sampler"):
            restore_sampler(trainer, self.round_sampler)
            self.replay_cap = message.get("replay_microbatches")
            print(f"Peer sampler restored for identical-data replay "
                  f"(cap={self.replay_cap})", flush=True)
        if diagnostics is not None:
            diagnostics.emit("weights_synced", updated=message["updated"], new_weight_version=trainer.step)
        self.channel.send(event="synced")
        return {}


def peer_main(fd, setup_path):
    from .backend import BACKEND_NAME, xp
    from .config import ModelConfig
    from .model.decoder_lm import DecoderLanguageModel
    from .train_extended import ExtendedTrainer
    setup = json.loads(Path(setup_path).read_text())
    config = setup["config"]
    if BACKEND_NAME == "cupy":
        xp.cuda.Device(int(config.get("device", 0))).use()
        runtime = "hip" if xp.cuda.runtime.is_hip else "cuda"
        if runtime != config.get("expected_runtime", "hip"):
            raise RuntimeError("unexpected peer GPU runtime")
    model_config = ModelConfig(**setup["model_config"])
    model = DecoderLanguageModel(model_config, rng_seed=104729, dtype=model_config.dtype)
    trainer = ExtendedTrainer(model, **setup["trainer_kwargs"])
    if setup["sampler_state"]:
        restore_sampler(trainer, setup["sampler_state"])
    trainer.step = setup["step"]
    diagnostics = _start_diagnostics(trainer, setup["diagnostics"], _diagnostic_role("peer"))
    if diagnostics is not None:
        print(f"Per-GPU diagnostic log: {diagnostics.path}", flush=True)
    channel = Channel(socket.socket(fileno=fd))
    buffers = {name: np.memmap(Path(setup["directory"]) / name, dtype=np.float32, mode="r+", shape=(setup["count"],))
               for name in ("weights", "peer-gradient")}
    schema = [(p.name, list(p.shape)) for p in model.parameters()]
    channel.send(event="ready", schema=schema)
    hooks = _PeerHooks(channel, setup, buffers)
    trainer.accumulation_peer = hooks
    while True:
        message = channel.receive(float(config.get("timeout_s", 1800)))
        command = message["command"]
        if command == "stop":
            if diagnostics is not None:
                diagnostics.emit("session_end", trainer_step=trainer.step)
                diagnostics.close()
            return
        if command == "initialize":
            _copy(list(model.parameters()), buffers["weights"], "data", False)
            model.refresh_compute_buffers()
            channel.send(event="synced")
        elif command == "warmup":
            warmup(trainer, message["count"])
            channel.send(event="warmed")
        elif command == "begin":
            if message["step"] != trainer.step:
                raise ValueError("owner/peer step mismatch")
            with diagnostics.stage("train_step") if diagnostics else nullcontext():
                trainer.train_step()
        elif command == "state":
            channel.send(event="state", sampler_state=sampler_state(trainer))
        else:
            raise RuntimeError(f"unexpected peer command: {command}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--setup", required=True)
    args = parser.parse_args()
    peer_main(args.fd, args.setup)
