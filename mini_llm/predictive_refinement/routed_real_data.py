"""Q4: sample the original EOS-bounded production mixture for refinement.

The production sampler returns causal SOURCE and next-token LABEL windows.
Refinement instead predicts the label tokens themselves.  The first source
working token must therefore become the most recent *historical* token, and
all subsequent labels form the real refinement window. This is a one-token
sliding causal alignment, not a change to document selection probabilities.
"""
from __future__ import annotations

import numpy as np

from mini_llm.data.token_shards import create_hierarchical_memory_minibatch
from mini_llm.predictive_refinement.real_data import PackedRefinementDocuments


def align_next_token_targets(source_inputs, target_labels, metadata, *, memory_length,
                             pad_token_id=0):
    """Return [D real/padded history | T label tokens] and their valid masks.

    Before: source [h[-D:], x_0, x_1, ...], labels [x_1, x_2, ...].
    After:  source [h[-D+1:], x_0 | x_1, x_2, ...].
    The history left boundary moves by one (or remains zero when full).
    """
    src = np.asarray(source_inputs)
    labels = np.asarray(target_labels)
    D = int(memory_length)
    if src.ndim != 2 or labels.ndim != 2 or src.shape[0] != labels.shape[0] or src.shape[1] != D + labels.shape[1]:
        raise ValueError('invalid production source/label shapes')
    lengths = np.asarray(metadata['target_valid_lengths'], dtype=np.int64)
    starts = np.asarray(metadata['history_valid_starts'], dtype=np.int64)
    B, T = labels.shape
    if lengths.shape != (B,) or starts.shape != (B,) or np.any(lengths < 0) or np.any(lengths > T):
        raise ValueError('invalid production valid lengths/history starts')
    if np.any(starts < 0) or np.any(starts > D):
        raise ValueError('invalid production history boundaries')
    # Only a real first input can be reclassified as historical context.
    # Documents with zero prediction pairs are retained in the sampling
    # distribution but carry no training supervision.
    shifted_starts = np.where(lengths > 0, np.maximum(0, starts - 1), D)
    shifted_history = src[:, 1:D+1].copy()
    shifted_labels = labels.copy()
    valid = np.arange(T)[None, :] < lengths[:, None]
    shifted_labels[~valid] = pad_token_id
    shifted_history[np.arange(D)[None, :] < shifted_starts[:, None]] = pad_token_id
    source = np.concatenate((shifted_history, shifted_labels), axis=1)
    meta = {
        'history_valid_starts': shifted_starts,
        'target_valid_lengths': lengths,
        'router_stochastic': False,
        'causal_prefix_query': True,
        'causal_masked_target_refinement': True,
        'use_retrieval': True,
    }
    return source, valid, meta


class PackedRoutedRefinementDocuments(PackedRefinementDocuments):
    """Mixture selection and EOS document indexing from 0079I, window from 0078E."""

    def sample_routed(self, rng, *, batch_size, memory_length, target_length,
                      min_history_tokens=0):
        if min(batch_size, memory_length, target_length) <= 0:
            raise ValueError('positive sample shape required')
        sources = []
        masks = []
        starts = []
        lengths = []
        observations = []
        for _ in range(batch_size):
            source_i = int(rng.choice(len(self.sources), p=self.source_probs))
            name, _, paths = self.sources[source_i]
            shard_i = int(rng.choice(len(paths), p=self.shard_weights[source_i]))
            tokens, index = self._get_shard(paths[shard_i])
            src, labels, prod = create_hierarchical_memory_minibatch(
                tokens, 1, memory_length, target_length, rng,
                eos_token_id=self.eos, document_index=index,
                pad_token_id=self.eos, document_aware=True,
                return_metadata=True, min_history_tokens=int(min_history_tokens),
                sampling_policy='document_first',
            )
            shifted, valid, meta = align_next_token_targets(
                src, labels, prod, memory_length=memory_length,
                pad_token_id=self.eos,
            )
            sources.append(shifted[0])
            masks.append(valid[0])
            starts.append(int(meta['history_valid_starts'][0]))
            lengths.append(int(meta['target_valid_lengths'][0]))
            observations.append({
                'source': name,
                'sampled_document_span_tokens': int(prod['document_ends'][0] - prod['document_starts'][0]),
                'sampled_fragment_tokens': int(lengths[-1]),
                'route_eligible_history_tokens': memory_length - starts[-1],
            })
        metadata = dict(history_valid_starts=np.asarray(starts, dtype=np.int64),
                        target_valid_lengths=np.asarray(lengths, dtype=np.int64),
                        router_stochastic=False, causal_prefix_query=True,
                        causal_masked_target_refinement=True, use_retrieval=True)
        return np.stack(sources), np.stack(masks), metadata, observations
