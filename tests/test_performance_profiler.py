from mini_llm.performance_profiler import (
    PERFORMANCE_PROFILER,
    configure_performance_profiler,
    performance_report,
    performance_scope,
)


def test_profiler_is_noop_when_disabled():
    configure_performance_profiler(False, reset=True)
    with performance_scope("disabled.region"):
        sum(range(10))
    assert PERFORMANCE_PROFILER.rows() == []


def test_profiler_records_and_reports_regions():
    configure_performance_profiler(True, reset=True)
    with performance_scope("outer"):
        with performance_scope("inner"):
            sum(i * i for i in range(1000))
    configure_performance_profiler(False)

    rows = {row["name"]: row for row in PERFORMANCE_PROFILER.rows()}
    assert set(rows) == {"outer", "inner"}
    assert rows["outer"]["count"] == 1
    assert rows["inner"]["count"] == 1
    assert rows["outer"]["total_s"] >= 0.0
    assert rows["inner"]["total_s"] >= 0.0

    report = performance_report("test profile")
    assert "test profile" in report
    assert "outer" in report
    assert "inner" in report


def test_moe_detail_profiler_records_host_submit_time_on_numpy():
    configure_performance_profiler(True, reset=True)
    old = PERFORMANCE_PROFILER.moe_detail_enabled
    PERFORMANCE_PROFILER.moe_detail_enabled = True
    try:
        from mini_llm.performance_profiler import moe_detail_scope

        with moe_detail_scope("moe.test"):
            sum(i * i for i in range(100))
    finally:
        PERFORMANCE_PROFILER.moe_detail_enabled = old
        configure_performance_profiler(False)

    rows = {row["name"]: row for row in PERFORMANCE_PROFILER.rows()}
    assert "moe.test.host" in rows
    assert rows["moe.test.host"]["count"] == 1
    assert rows["moe.test.host"]["total_s"] >= 0.0
