"""Opt-in 0080AA: compare real live training microbatches against evaluation.

The audit reads the *same* inputs, targets, loss mask, and routing metadata
immediately after the production forward/backward pass, before optimizer update.
It aggregates per-source and per-horizon token-weighted NLL, rather than
comparing different random train and validation windows. Disabled by default.
"""

from collections import defaultdict

import numpy as np
from mini_llm.backend import asnumpy


class LiveBlockProposalAudit:
    def __init__(self, k):
        self.k = int(k)
        self.sources = defaultdict(self._empty)
        self.train_counts = np.zeros(self.k, dtype=np.float64)
        self.train_proposal = np.zeros(self.k, dtype=np.float64)
        self.train_final = np.zeros(self.k, dtype=np.float64)
        self.maximum_ce_error = 0.0
        self.maximum_horizon_error = 0.0
        self.microbatches = 0

    def _empty(self):
        return {
            "microbatches": 0,
            "retrieval_microbatches": 0,
            "length_sum": 0,
            "length_count": 0,
            "full_length_count": 0,
            "counts": np.zeros(self.k, dtype=np.float64),
            "train_proposal": np.zeros(self.k, dtype=np.float64),
            "train_final": np.zeros(self.k, dtype=np.float64),
            "eval_proposal": np.zeros(self.k, dtype=np.float64),
            "eval_final": np.zeros(self.k, dtype=np.float64),
        }

    def _arrays(self, metrics):
        counts = np.asarray(asnumpy(metrics["position_counts"]), dtype=np.float64)
        proposal = np.asarray(asnumpy(metrics["proposal_position_ce_sums"]), dtype=np.float64)
        final = np.asarray(asnumpy(metrics["final_position_ce_sums"]), dtype=np.float64)
        if counts.shape != (self.k,) or proposal.shape != counts.shape or final.shape != counts.shape:
            raise ValueError("unexpected K-horizon metric shape")
        total = int(metrics["unique_supervised_tokens"])
        if not np.isclose(counts.sum(), total, atol=0, rtol=0):
            raise ValueError("per-horizon counts do not match supervised-token count")
        # Check the two independent representations of the same loss that
        # production accumulates and logs. This also catches a bad reduction.
        for key, value in (("proposal_ce", proposal), ("final_ce", final)):
            ce = float(asnumpy(metrics[key]))
            if abs(float(value.sum()) / total - ce) > 0.005:
                raise ValueError(f"inconsistent {key}: per-horizon sums versus overall CE")
        return counts, proposal, final

    def add(self, source, training_metrics, evaluation_metrics, metadata=None,
            full_target_length=None):
        tc, tp, tf = self._arrays(training_metrics)
        ec, ep, ef = self._arrays(evaluation_metrics)
        if not np.array_equal(tc, ec):
            raise ValueError("live train/eval target counts differ on the same batch")
        stats = self.sources[str(source)]
        stats["microbatches"] += 1
        if metadata is not None:
            stats["retrieval_microbatches"] += int(bool(metadata.get("use_retrieval", False)))
            if "target_valid_lengths" in metadata:
                lengths = np.asarray(metadata["target_valid_lengths"], dtype=np.int64)
                stats["length_sum"] += int(lengths.sum())
                stats["length_count"] += int(lengths.size)
                if full_target_length is not None:
                    stats["full_length_count"] += int(np.sum(lengths == int(full_target_length)))
        for key, value in (("counts", tc), ("train_proposal", tp),
                           ("train_final", tf), ("eval_proposal", ep),
                           ("eval_final", ef)):
            stats[key] += value
        self.train_counts += tc
        self.train_proposal += tp
        self.train_final += tf
        valid = tc > 0
        if np.any(valid):
            per_h = np.concatenate(((tp[valid]-ep[valid])/tc[valid],
                                    (tf[valid]-ef[valid])/tc[valid]))
            self.maximum_horizon_error = max(self.maximum_horizon_error,
                                             float(np.max(np.abs(per_h))))
        n = float(tc.sum())
        self.maximum_ce_error = max(self.maximum_ce_error,
                                    abs(float(tp.sum()-ep.sum())/n),
                                    abs(float(tf.sum()-ef.sum())/n))
        self.microbatches += 1

    def check_production_accumulator(self, metrics):
        """Compare the real training logger's values with this CPU audit."""
        counts = np.asarray(asnumpy(metrics["position_counts"]), dtype=np.float64)
        proposal = np.asarray(asnumpy(metrics["proposal_position_ce_sums"]), dtype=np.float64)
        final = np.asarray(asnumpy(metrics["final_position_ce_sums"]), dtype=np.float64)
        return max(float(np.max(np.abs(counts-self.train_counts))),
                   float(np.max(np.abs(proposal-self.train_proposal))),
                   float(np.max(np.abs(final-self.train_final))))

    def format_report(self, step, original_metrics):
        lines = [f"0080AA LIVE TRAIN-EVAL AUDIT step={step} microbatches={self.microbatches}",
                 f"  exact-microbatch max |train-eval CE|={self.maximum_ce_error:.6f}; "
                 f"max |per-horizon CE|={self.maximum_horizon_error:.6f}",
                 f"  trainer per-horizon accumulator max |NLL/count difference|="
                 f"{self.check_production_accumulator(original_metrics):.6f}"]
        for name, data in sorted(self.sources.items()):
            info = (f"  {name}: microbatches={data['microbatches']} "
                    f"targets={int(data['counts'].sum()):,} "
                    f"retrieval={data['retrieval_microbatches']}/{data['microbatches']}")
            if data['length_count']:
                info += (f" mean_target_length={data['length_sum']/data['length_count']:.0f} "
                         f"full_windows={data['full_length_count']}/{data['length_count']}")
            lines.append(info)
            for h, n in enumerate(data["counts"]):
                if not n:
                    continue
                tp, tf = data["train_proposal"][h]/n, data["train_final"][h]/n
                ep, ef = data["eval_proposal"][h]/n, data["eval_final"][h]/n
                lines.append(f"    h{h+1}: training P={tp:.4f} F={tf:.4f} gain={tp-tf:+.4f} "
                             f"| same-batch eval P={ep:.4f} F={ef:.4f} gain={ep-ef:+.4f}")
        return "\n".join(lines)
