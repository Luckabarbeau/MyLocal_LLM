"""0080P1: frozen two-stage checkpoint adapter for the existing prompt benchmark.

Supports the 0080I small CPU research checkpoint layout.  The big-model
0080N/0080O production checkpoint format requires a separately verified loader;
this adapter deliberately refuses to misinterpret it as conventional AR.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, BACKEND_NAME, synchronize
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.inference_model import InferenceModel
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
from mini_llm.block_proposal.checkpoint import ARCHITECTURE, load_two_stage_checkpoint
from mini_llm.block_proposal.generation import SequentialCommitBlockGenerator
from mini_llm.predictive_refinement.real_data import sha256
from mini_llm.tokenizer.tokenizer import FastBPETokenizer


def load_0080i_block_checkpoint(checkpoint: Path, tokenizer_path: Path):
    """Load exact 0080I weights/config without opening the original train shards."""
    from examples.evaluate_block_proposal_ablations_0080j import _reconstruct

    checkpoint = Path(checkpoint).expanduser().resolve()
    tokenizer_path = Path(tokenizer_path).expanduser().resolve()
    state = json.loads((checkpoint / 'training_state.json').read_text(encoding='utf-8'))
    if not isinstance(state.get('architecture'), dict) or state['architecture'].get('type') != ARCHITECTURE:
        raise ValueError('expected a two-stage 0080I research checkpoint; conventional AR is not supported here')
    signature = state.get('data_provenance')
    if not isinstance(signature, dict) or signature.get('version') != '0080I-v1':
        raise ValueError('0080P1 needs an 0080I-v1 checkpoint; production 500M format is not yet supported')
    tokenizer = FastBPETokenizer.load(str(tokenizer_path))
    if state.get('tokenizer_hash') != sha256(tokenizer_path):
        raise ValueError('checkpoint tokenizer checksum differs from the selected tokenizer')
    a, cfg = _reconstruct(signature, len(tokenizer))
    main = DecoderLanguageModel(cfg, rng_seed=a.seed, dtype=a.dtype)
    head = HorizonProposalHead(main, a.block_size, hidden_width=a.proposal_width, rng_seed=a.seed+1)
    decoder = BlockPrefixDecoder(main, a.block_size, width=a.decoder_width,
        n_layers=a.decoder_layers, n_heads=a.decoder_heads, rng_seed=a.seed+2)
    net = TwoStageBlockModel(head, decoder, lambda_proposal=a.lambda_proposal)
    # Check both the complete exact architecture fingerprint and tokenizer hash.
    load_two_stage_checkpoint(checkpoint, net, optimizer=None,
        expected_tokenizer_hash=sha256(tokenizer_path),
        expected_data_provenance=signature)
    if int(state['step']) < 0:
        raise ValueError('invalid training step')
    return net, tokenizer, dict(step=int(state['step']), context=int(cfg.memory_context.memory_length),
        model_dtype=cfg.dtype, backend=BACKEND_NAME, tokenizer_path=str(tokenizer_path),
        architecture='0080I_two_stage', block_size=a.block_size)


class BlockProposalTextGenerator:
    """Same generate() contract as TextGenerator for benchmarks/evaluate_generation.py."""

    def __init__(self, net, tokenizer, *, max_context, commit_stride=None):
        self.net = net
        self.tokenizer = tokenizer
        self.max_context = int(max_context)
        self.model = net.proposal_head.model
        horizon = int(net.proposal_head.block_size)
        self.commit_stride = horizon if commit_stride is None else int(commit_stride)
        if not 1 <= self.commit_stride <= horizon:
            raise ValueError(f'commit_stride must be between 1 and K={horizon}')
        mc = self.model.config.memory_context
        max_trained = int(mc.memory_length) if mc.enabled else int(self.model.config.context_length)
        if self.max_context < 1 or self.max_context > max_trained:
            raise ValueError(f'context {self.max_context} outside checkpoint range [1,{max_trained}]')
        self.inference = InferenceModel(self.model.config, dtype=self.model.dtype)
        self.inference.set_weights(self.model)
        self.eos_id = self.tokenizer._tokenizer.token_to_id(self.tokenizer.eos_token)
        if self.eos_id is None:
            raise ValueError('tokenizer EOS token missing')

    def generate(self, prompt, max_new_tokens=50, strategy='greedy', temperature=1.,
                 top_k=None, top_p=None, seed=None, log_speed=False,
                 use_kv_cache=True, return_token_ids=False):
        from inference import greedy_decode, temperature_sample, top_k_sample, top_p_sample

        if not use_kv_cache:
            raise NotImplementedError('block-proposal reference requires causal KV cache')
        if int(max_new_tokens) <= 0:
            raise ValueError('max_new_tokens must be positive')
        if seed is not None:
            np.random.seed(seed)
            xp.random.seed(seed)
        ids = list(self.tokenizer.encode(prompt))
        if not ids:
            raise ValueError('prompt tokenization produced no tokens; supply non-empty prompt')
        if len(ids) >= self.max_context:
            raise ValueError('prompt fills the entire context: use a shorter prompt or larger max context')
        ids = ids[-self.max_context:]
        capacity = min(self.max_context, len(ids) + int(max_new_tokens))
        state = self.inference.create_generation_state(batch_size=1, max_length=capacity)

        def chooser(scores):
            if strategy == 'greedy':
                return greedy_decode(scores)
            if strategy == 'temperature':
                return temperature_sample(scores, temperature)
            if strategy == 'top-k':
                return top_k_sample(scores, top_k, temperature)
            if strategy == 'top-p':
                return top_p_sample(scores, top_p, temperature)
            raise ValueError(f'unsupported sampling strategy: {strategy}')

        if log_speed:
            synchronize()
        start = time.perf_counter()
        self.inference.prefill(xp.asarray([ids], dtype=xp.int32), state)
        if log_speed:
            synchronize()
        prefill = time.perf_counter() - start
        gen = SequentialCommitBlockGenerator(self.inference, self.net.proposal_head,
            self.net.decoder, chooser=chooser, eos_token_id=self.eos_id,
            commit_stride=self.commit_stride)
        output = gen.generate(state, min(int(max_new_tokens), capacity-len(ids)))
        if log_speed:
            synchronize()
        elapsed = time.perf_counter() - start
        tokens = output['tokens']
        text = self.tokenizer.decode(tokens)
        timing = {'tokens_per_sec': len(tokens)/elapsed if elapsed else 0.,
                  'total_time_s': elapsed, 'prefill_time_s': prefill,
                  'block_proposal_implementation': output['implementation'],
                  'proposal_blocks': output['proposal_blocks'],
                  'causal_main_updates': output['causal_main_updates'],
                  'commit_stride': output['commit_stride'],
                  'proposal_horizon': output['proposal_horizon'],
                  'benchmark_warning': 'reference sequential main-KV updates; not a block speed measurement'}
        if return_token_ids:
            timing.update(generated_token_ids=tokens, eos_token_id=self.eos_id)
        return text, timing
