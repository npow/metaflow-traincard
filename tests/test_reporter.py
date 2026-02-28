"""Tests for the core Reporter SDK."""

import json
import math
import os
import signal
import tempfile
import threading
import time
from pathlib import Path

import pytest

from metaflow_traincard.reporter import Reporter, _MAX_LOG_LINES


# ─── metric() ────────────────────────────────────────────────────────

class TestMetric:
    def test_records_single_metric(self, reporter):
        reporter.metric("loss", 1.5, step=0)
        state = reporter.get_state()
        assert "loss" in state["metrics"]
        assert state["metrics"]["loss"] == [{"step": 0, "value": 1.5}]

    def test_records_multiple_metrics(self, reporter):
        reporter.metric("loss", 1.5, step=0)
        reporter.metric("lr", 1e-4, step=0)
        state = reporter.get_state()
        assert "loss" in state["metrics"]
        assert "lr" in state["metrics"]

    def test_auto_increments_step(self, reporter):
        reporter.metric("loss", 1.5, step=10)
        assert reporter.get_state()["step"] == 10
        reporter.metric("loss", 1.4, step=20)
        assert reporter.get_state()["step"] == 20

    def test_step_monotonic(self, reporter):
        reporter.metric("loss", 1.5, step=50)
        reporter.metric("loss", 1.4, step=10)  # Earlier step
        assert reporter.get_state()["step"] == 50  # Should not decrease

    def test_deduplicates_same_step(self, reporter):
        reporter.metric("loss", 1.5, step=0)
        reporter.metric("loss", 1.5, step=0)  # Exact duplicate
        state = reporter.get_state()
        assert len(state["metrics"]["loss"]) == 1

    def test_appends_different_steps(self, reporter):
        for i in range(5):
            reporter.metric("loss", float(i), step=i)
        assert len(reporter.get_state()["metrics"]["loss"]) == 5

    def test_value_cast_to_float(self, reporter):
        reporter.metric("loss", 2, step=0)  # int input
        val = reporter.get_state()["metrics"]["loss"][0]["value"]
        assert isinstance(val, float)
        assert val == 2.0

    def test_non_finite_values_stored(self, reporter):
        """inf/nan values should be stored (Chart.js handles them as gaps)."""
        reporter.metric("loss", float("nan"), step=0)
        state = reporter.get_state()
        assert math.isnan(state["metrics"]["loss"][0]["value"])

    def test_rank_nonzero_skips(self, tmp_path):
        r = Reporter(
            output_dir=str(tmp_path / "tc"),
            flush_interval=9999,
            rank=1,
            world_size=4,
        )
        r.metric("loss", 1.0, step=0)
        state = r.get_state()
        assert state["metrics"] == {}  # Non-main rank is silent
        r.finish()

    def test_tags_accepted(self, reporter):
        reporter.metric("loss", 1.5, step=0, tags={"split": "train"})
        # Tags are not stored in state for now, just should not raise
        assert "loss" in reporter.get_state()["metrics"]

    def test_thread_safety(self, reporter):
        """Concurrent metric calls must not corrupt state."""
        errors = []

        def write_metrics(start):
            try:
                for i in range(50):
                    reporter.metric("loss", float(i), step=start + i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_metrics, args=(i * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Thread errors: {errors}"


# ─── log() ───────────────────────────────────────────────────────────

class TestLog:
    def test_records_log(self, reporter):
        reporter.log("hello")
        logs = reporter.get_state()["logs"]
        assert len(logs) == 1
        assert logs[0]["line"] == "hello"

    def test_log_has_timestamp(self, reporter):
        reporter.log("test")
        assert reporter.get_state()["logs"][0]["time"] > 0

    def test_log_level_default_info(self, reporter):
        reporter.log("test")
        assert reporter.get_state()["logs"][0]["level"] == "info"

    def test_log_level_custom(self, reporter):
        reporter.log("bad thing", level="error")
        assert reporter.get_state()["logs"][0]["level"] == "error"

    def test_log_capped_at_max(self, reporter):
        for i in range(_MAX_LOG_LINES + 100):
            reporter.log(f"line {i}")
        logs = reporter.get_state()["logs"]
        assert len(logs) <= _MAX_LOG_LINES
        # Most recent lines should be kept
        assert logs[-1]["line"] == f"line {_MAX_LOG_LINES + 99}"


# ─── phase() ─────────────────────────────────────────────────────────

class TestPhase:
    def test_sets_phase(self, reporter):
        reporter.phase("eval")
        assert reporter.get_state()["phase"] == "eval"

    def test_phase_transitions(self, reporter):
        for p in ["train", "eval", "save", "done"]:
            reporter.phase(p)
            assert reporter.get_state()["phase"] == p


# ─── checkpoint() ────────────────────────────────────────────────────

class TestCheckpoint:
    def test_records_checkpoint(self, reporter):
        reporter.metric("loss", 1.0, step=50)
        reporter.checkpoint("/tmp/ckpt-50")
        ckpts = reporter.get_state()["checkpoints"]
        assert len(ckpts) == 1
        assert ckpts[0]["path"] == "/tmp/ckpt-50"
        assert ckpts[0]["step"] == 50

    def test_checkpoint_metadata(self, reporter):
        reporter.checkpoint("/tmp/ckpt", metadata={"epoch": 3, "eval_loss": 1.2})
        meta = reporter.get_state()["checkpoints"][0]["metadata"]
        assert meta["epoch"] == 3
        assert meta["eval_loss"] == 1.2

    def test_checkpoint_has_timestamp(self, reporter):
        reporter.checkpoint("/tmp/ckpt")
        assert reporter.get_state()["checkpoints"][0]["time"] > 0

    def test_multiple_checkpoints(self, reporter):
        for i in range(3):
            reporter.metric("loss", float(i), step=i * 100)
            reporter.checkpoint(f"/tmp/ckpt-{i}")
        assert len(reporter.get_state()["checkpoints"]) == 3


# ─── system() ────────────────────────────────────────────────────────

class TestSystem:
    def test_records_system_stats(self, reporter):
        stats = {"gpu_utilization": [80.0], "cpu_percent": 30.0}
        reporter.system(stats)
        assert reporter.get_state()["system"] == stats

    def test_overwrites_previous(self, reporter):
        reporter.system({"cpu_percent": 20.0})
        reporter.system({"cpu_percent": 80.0})
        assert reporter.get_state()["system"]["cpu_percent"] == 80.0


# ─── heartbeat() ─────────────────────────────────────────────────────

class TestHeartbeat:
    def test_updates_timestamp(self, reporter):
        old = reporter.get_state()["last_heartbeat"]
        time.sleep(0.05)
        reporter.heartbeat()
        new = reporter.get_state()["last_heartbeat"]
        assert new >= old

    def test_clears_stall_flag(self, reporter, tmp_path):
        r = Reporter(
            output_dir=str(tmp_path / "tc"),
            flush_interval=9999,
            stall_timeout=0,  # Immediate stall
        )
        r.metric("loss", 1.0, step=0)
        r.phase("train")
        r._flush()  # Triggers stall detection
        r._state["stalled"] = True
        r.heartbeat()
        assert r.get_state()["stalled"] is False
        r.finish()


# ─── failure() ───────────────────────────────────────────────────────

class TestFailure:
    def test_records_failure(self, reporter):
        reporter.failure("RuntimeError", "CUDA OOM", "traceback here")
        f = reporter.get_state()["failure"]
        assert f["type"] == "RuntimeError"
        assert f["message"] == "CUDA OOM"
        assert f["traceback"] == "traceback here"

    def test_oom_detection(self, reporter):
        reporter.failure("RuntimeError", "CUDA out of memory. Tried to allocate 8 GiB")
        assert reporter.get_state()["failure"]["oom_suspected"] is True

    def test_no_oom_false_positive(self, reporter):
        reporter.failure("ValueError", "Invalid input shape")
        assert reporter.get_state()["failure"]["oom_suspected"] is False


# ─── epoch() ─────────────────────────────────────────────────────────

class TestEpoch:
    def test_sets_epoch(self, reporter):
        reporter.epoch(3)
        assert reporter.get_state()["epoch"] == 3


# ─── finish() ────────────────────────────────────────────────────────

class TestFinish:
    def test_sets_done_phase(self, reporter):
        reporter.finish()
        assert reporter.get_state()["phase"] == "done"

    def test_writes_latest_json(self, reporter):
        reporter.metric("loss", 1.0, step=0)
        reporter.finish()
        latest = reporter.output_dir / "latest.json"
        assert latest.exists()
        data = json.loads(latest.read_text())
        assert "metrics" in data

    def test_latest_json_valid(self, reporter):
        reporter.metric("loss", 1.5, step=10)
        reporter.log("hello")
        reporter.checkpoint("/tmp/ckpt")
        reporter.finish()
        data = json.loads((reporter.output_dir / "latest.json").read_text())
        assert data["metrics"]["loss"][0]["value"] == 1.5
        assert data["logs"][0]["line"] == "hello"
        assert data["checkpoints"][0]["path"] == "/tmp/ckpt"


# ─── Flush / file writing ─────────────────────────────────────────────

class TestFlush:
    def test_atomic_write(self, reporter):
        """Tmp file should not persist after flush."""
        reporter.metric("loss", 1.0, step=0)
        reporter._flush()
        tmp = reporter.output_dir / "latest.tmp"
        assert not tmp.exists()

    def test_events_jsonl_written(self, reporter):
        reporter.metric("loss", 1.5, step=5)
        reporter.log("a line")
        reporter.phase("eval")
        time.sleep(0.1)  # Let writes flush
        events_path = reporter.output_dir / "events.jsonl"
        assert events_path.exists()
        lines = events_path.read_text().strip().splitlines()
        event_types = [json.loads(l)["type"] for l in lines]
        assert "metric" in event_types
        assert "phase" in event_types

    def test_checkpoints_json_written(self, reporter):
        reporter.checkpoint("/tmp/ckpt")
        ckpt_path = reporter.output_dir / "checkpoints.json"
        assert ckpt_path.exists()
        data = json.loads(ckpt_path.read_text())
        assert data[0]["path"] == "/tmp/ckpt"


# ─── Resume continuity ───────────────────────────────────────────────

class TestResume:
    def test_resume_loads_prior_metrics(self, tmp_path):
        # First run
        r1 = Reporter(output_dir=str(tmp_path / "tc"), flush_interval=9999)
        for i in range(5):
            r1.metric("loss", float(i), step=i)
        r1.finish()

        # Second run (same dir)
        r2 = Reporter(output_dir=str(tmp_path / "tc"), flush_interval=9999)
        state = r2.get_state()
        assert "loss" in state["metrics"]
        assert state["restart_count"] == 1
        r2.finish()

    def test_resume_increments_restart_count(self, tmp_path):
        # Run 1: no prior state → restart_count = 0
        # Run 2: sees run-1 state → restart_count = 1
        # Run 3: sees run-2 state → restart_count = 2
        for run_idx in range(3):
            r = Reporter(output_dir=str(tmp_path / "tc"), flush_interval=9999)
            r.finish()
            assert r.get_state()["restart_count"] == run_idx


# ─── Stall detection ─────────────────────────────────────────────────

class TestStall:
    def test_stall_flag_set_after_timeout(self, tmp_path):
        r = Reporter(
            output_dir=str(tmp_path / "tc"),
            flush_interval=9999,
            stall_timeout=0,  # Immediate
        )
        r.metric("loss", 1.0, step=0)
        r.phase("train")
        time.sleep(0.01)
        r._flush()
        assert r.get_state()["stalled"] is True
        r.finish()

    def test_no_stall_in_init_phase(self, tmp_path):
        r = Reporter(
            output_dir=str(tmp_path / "tc"),
            flush_interval=9999,
            stall_timeout=0,
        )
        r.phase("init")
        r._flush()
        assert r.get_state()["stalled"] is False
        r.finish()

    def test_no_stall_in_done_phase(self, tmp_path):
        r = Reporter(
            output_dir=str(tmp_path / "tc"),
            flush_interval=9999,
            stall_timeout=0,
        )
        r.phase("done")
        r._flush()
        assert r.get_state()["stalled"] is False
        r.finish()


# ─── SIGTERM handler ─────────────────────────────────────────────────

class TestSigterm:
    def test_sigterm_flushes_state(self, reporter):
        reporter.metric("loss", 1.5, step=10)
        # Simulate SIGTERM
        reporter._on_sigterm(signal.SIGTERM, None)
        latest = reporter.output_dir / "latest.json"
        assert latest.exists()
        data = json.loads(latest.read_text())
        assert "loss" in data["metrics"]
