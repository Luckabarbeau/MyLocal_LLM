"""0085J: grouped local recomputation accepts sliced Q with packed dcontext."""

from mini_llm.ops.cublas_grouped import RowMajorGemmProblem
from mini_llm.ops.indexed_attention import _local_grouped_recompute_input_groups


def _gemm(lda, ldb, i):
    return RowMajorGemmProblem(
        1000 + i * 100, 2000 + i * 100, 3000 + i * 100,
        16, 48, 64, lda, ldb, 48, False, True,
    )


def _signature(p):
    return (p.m, p.n, p.k, p.lda, p.ldb, p.ldc, p.trans_a, p.trans_b)


def _check_uniform(groups):
    for group in groups:
        assert len({_signature(p) for p in group.problems}) == 1


def test_sliced_q_compact_dcontext_separates_two_uniform_groups():
    # Head-selection slice (15 heads) retains original 16-head token stride.
    # Backward dcontext is compacted to the selected 15 heads.
    score = [_gemm(16 * 64, 4 * 64, i) for i in range(6)]
    dprobs = [_gemm(15 * 64, 4 * 64, i + 100) for i in range(6)]
    groups = _local_grouped_recompute_input_groups(
        score, dprobs, q_row=1024, k_row=256, dc_row=960, v_row=256
    )
    assert len(groups) == 2
    assert len(groups[0].problems) == len(groups[1].problems) == 6
    assert tuple(groups[0].problems) == tuple(score)
    assert tuple(groups[1].problems) == tuple(dprobs)
    _check_uniform(groups)


def test_matched_layouts_preserve_single_group_fast_path():
    score = [_gemm(960, 256, i) for i in range(6)]
    dprobs = [_gemm(960, 256, i + 100) for i in range(6)]
    groups = _local_grouped_recompute_input_groups(
        score, dprobs, q_row=960, k_row=256, dc_row=960, v_row=256,
    )
    assert len(groups) == 1
    assert len(groups[0].problems) == 12
    assert groups[0].problems[0] is score[0]
    assert groups[0].problems[1] is dprobs[0]
    _check_uniform(groups)


def test_strided_v_also_requires_separate_groups():
    score = [_gemm(960, 256, i) for i in range(6)]
    dprobs = [_gemm(960, 1024, i + 100) for i in range(6)]
    groups = _local_grouped_recompute_input_groups(
        score, dprobs, q_row=960, k_row=256, dc_row=960, v_row=1024
    )
    assert len(groups) == 2
    _check_uniform(groups)


def test_scores_only_recompute_remains_single_group():
    score = [_gemm(1024, 256, i) for i in range(6)]
    groups = _local_grouped_recompute_input_groups(
        score, [], q_row=1024, k_row=256, dc_row=0, v_row=0
    )
    assert len(groups) == 1
    assert tuple(groups[0].problems) == tuple(score)
    _check_uniform(groups)
