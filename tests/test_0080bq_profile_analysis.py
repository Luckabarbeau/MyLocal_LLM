"""Pure-CPU parser checks for 0080BQ read-only deep AR benchmark report."""
from benchmarks.analyze_deep_ar_profile_0080bq import parse_profile, summarize

SAMPLE = """some startup logging
Performance profile: optimizer step 1
-------------------------------------
region                                     total ms    calls       avg ms
train.model_forward                         501.000        2      250.500
block.attention.forward                     300.000       64        4.688
attention.local.forward                     220.000       64        3.438
train.model_backward                       1000.000        2      500.000
block.moe.backward                          180.000       64        2.812
moe.expert.gate_gemm.gpu                     90.000       64        1.406
moe.expert.gate_gemm.host                     5.000       64        0.078

GPU memory after profiled step: used=3.50 GiB
Profiled optimizer-step wall time: 2.000s
  target capacity tokens/s: 123
"""


def test_profile_parsed():
    parsed = parse_profile(SAMPLE)
    assert len(parsed) == 1
    report = summarize(parsed[0])
    assert report["step"] == 1
    assert report["wall_ms"] == 2000
    assert report["phases_ms"]["train.model_forward"] == 501
    assert report["unscoped_ms_estimate"] == 499
    assert report["block_components_ms"]["block.moe.backward"] == 180
    assert report["attention_subregions_ms"]["attention.local.forward"] == 220
    assert report["detail_ms"]["moe.expert.gate_gemm.gpu"] == 90


def test_missing_report():
    assert parse_profile("ordinary step log only") == []


def test_multiple_reports():
    assert len(parse_profile(SAMPLE + SAMPLE.replace("step 1", "step 2"))) == 2
