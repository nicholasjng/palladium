"""Benchmark artifacts survive unsupported sections and interrupted runs."""

import json

import pjrt_m1_benchmark as benchmark
import pytest


def test_checkpoint_survives_failure_and_records_next_section(tmp_path):
    path = tmp_path / "result.json"
    result = {"sections": {}}
    checkpoint = lambda value: benchmark._write_json(path, value)
    benchmark._record_section(result, "small", lambda: {"small_median_us": 20}, checkpoint)

    def unsupported():
        saved = json.loads(path.read_text())
        assert saved["small_median_us"] == 20
        assert saved["sections"]["sbi"]["status"] == "running"
        raise RuntimeError("UNIMPLEMENTED: test operation")

    benchmark._record_section(result, "sbi", unsupported, checkpoint)
    benchmark._record_section(result, "other", lambda: {"other_value": 7}, checkpoint)
    saved = json.loads(path.read_text())
    assert saved["small_median_us"] == 20 and saved["other_value"] == 7
    assert saved["sections"]["sbi"] == {
        "status": "error",
        "error_type": "RuntimeError",
        "message": "UNIMPLEMENTED: test operation",
    }


def test_interruption_leaves_running_section_and_previous_results(tmp_path):
    path = tmp_path / "result.json"
    result = {"sections": {}, "small_median_us": 20}

    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        benchmark._record_section(
            result, "large", interrupted, lambda value: benchmark._write_json(path, value)
        )
    saved = json.loads(path.read_text())
    assert saved["small_median_us"] == 20
    assert saved["sections"]["large"]["status"] == "running"
