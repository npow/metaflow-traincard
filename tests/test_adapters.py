"""Tests for framework adapters (HuggingFace, etc.)."""

import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from metaflow_traincard.adapters.huggingface import HFTrainCardCallback
from metaflow_traincard.reporter import Reporter


# ─── Helpers ─────────────────────────────────────────────────────────

def make_args(output_dir, **kwargs):
    ns = SimpleNamespace(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        world_size=1,
        local_rank=0,
        **kwargs,
    )
    return ns


def make_state(step=0, epoch=0.0):
    return SimpleNamespace(global_step=step, epoch=epoch)


def make_control():
    return SimpleNamespace()


# ─── Initialization ──────────────────────────────────────────────────

class TestHFInit:
    def test_creates_reporter_on_train_begin(self, tmp_path):
        cb = HFTrainCardCallback()
        cb.on_train_begin(make_args(str(tmp_path)), make_state(), make_control())
        assert cb.reporter is not None

    def test_uses_provided_reporter(self, reporter):
        cb = HFTrainCardCallback(reporter=reporter)
        cb.on_train_begin(make_args("/tmp/out"), make_state(), make_control())
        assert cb.reporter is reporter

    def test_output_dir_default(self, tmp_path):
        cb = HFTrainCardCallback()
        cb.on_train_begin(make_args(str(tmp_path)), make_state(), make_control())
        assert cb.reporter.output_dir == Path(str(tmp_path)) / "_traincard"

    def test_output_dir_override(self, tmp_path):
        out = str(tmp_path / "custom")
        cb = HFTrainCardCallback(output_dir=out)
        cb.on_train_begin(make_args(str(tmp_path)), make_state(), make_control())
        assert cb.reporter.output_dir == Path(out)


# ─── on_log ──────────────────────────────────────────────────────────

class TestOnLog:
    def test_records_metrics_from_logs(self, tmp_path):
        cb = HFTrainCardCallback()
        args = make_args(str(tmp_path))
        cb.on_train_begin(args, make_state(), make_control())
        cb.on_log(args, make_state(step=10), make_control(), logs={"loss": 1.5, "lr": 1e-4})
        state = cb.reporter.get_state()
        assert "loss" in state["metrics"]
        assert "lr" in state["metrics"]
        assert state["metrics"]["loss"][0]["value"] == 1.5

    def test_skips_non_numeric_values(self, tmp_path):
        cb = HFTrainCardCallback()
        args = make_args(str(tmp_path))
        cb.on_train_begin(args, make_state(), make_control())
        cb.on_log(args, make_state(step=5), make_control(), logs={"epoch": "not_a_number"})
        state = cb.reporter.get_state()
        assert state["metrics"] == {}

    def test_no_logs_is_noop(self, tmp_path):
        cb = HFTrainCardCallback()
        args = make_args(str(tmp_path))
        cb.on_train_begin(args, make_state(), make_control())
        cb.on_log(args, make_state(), make_control(), logs=None)
        assert cb.reporter.get_state()["metrics"] == {}

    def test_emits_log_line(self, tmp_path):
        cb = HFTrainCardCallback()
        args = make_args(str(tmp_path))
        cb.on_train_begin(args, make_state(), make_control())
        cb.on_log(args, make_state(step=20), make_control(), logs={"loss": 1.2})
        logs = cb.reporter.get_state()["logs"]
        assert any("loss" in l["line"] for l in logs)

    def test_uses_global_step(self, tmp_path):
        cb = HFTrainCardCallback()
        args = make_args(str(tmp_path))
        cb.on_train_begin(args, make_state(), make_control())
        cb.on_log(args, make_state(step=42), make_control(), logs={"loss": 1.0})
        state = cb.reporter.get_state()
        assert state["metrics"]["loss"][0]["step"] == 42


# ─── on_evaluate ─────────────────────────────────────────────────────

class TestOnEvaluate:
    def test_records_eval_metrics(self, tmp_path):
        cb = HFTrainCardCallback()
        args = make_args(str(tmp_path))
        cb.on_train_begin(args, make_state(), make_control())
        cb.on_evaluate(
            args, make_state(step=50), make_control(),
            metrics={"eval_loss": 1.3, "eval_accuracy": 0.85},
        )
        state = cb.reporter.get_state()
        assert "eval_loss" in state["metrics"]
        assert "eval_accuracy" in state["metrics"]

    def test_sets_eval_phase(self, tmp_path):
        cb = HFTrainCardCallback()
        args = make_args(str(tmp_path))
        cb.on_train_begin(args, make_state(), make_control())
        cb.on_evaluate(args, make_state(), make_control(), metrics={"eval_loss": 1.0})
        assert cb.reporter.get_state()["phase"] == "eval"


# ─── on_save ─────────────────────────────────────────────────────────

class TestOnSave:
    def test_records_checkpoint(self, tmp_path):
        cb = HFTrainCardCallback()
        args = make_args(str(tmp_path))
        cb.on_train_begin(args, make_state(), make_control())
        cb.on_save(args, make_state(step=100), make_control())
        state = cb.reporter.get_state()
        assert len(state["checkpoints"]) == 1
        assert "checkpoint-100" in state["checkpoints"][0]["path"]

    def test_checkpoint_has_step_metadata(self, tmp_path):
        cb = HFTrainCardCallback()
        args = make_args(str(tmp_path))
        cb.on_train_begin(args, make_state(), make_control())
        cb.on_save(args, make_state(step=200), make_control())
        meta = cb.reporter.get_state()["checkpoints"][0]["metadata"]
        assert meta["step"] == 200

    def test_phase_returns_to_train_after_save(self, tmp_path):
        cb = HFTrainCardCallback()
        args = make_args(str(tmp_path))
        cb.on_train_begin(args, make_state(), make_control())
        cb.on_save(args, make_state(step=100), make_control())
        assert cb.reporter.get_state()["phase"] == "train"


# ─── on_epoch_begin / end ────────────────────────────────────────────

class TestEpochEvents:
    def test_on_epoch_begin_updates_epoch(self, tmp_path):
        cb = HFTrainCardCallback()
        args = make_args(str(tmp_path))
        cb.on_train_begin(args, make_state(), make_control())
        cb.on_epoch_begin(args, make_state(step=0, epoch=2.0), make_control())
        assert cb.reporter.get_state()["epoch"] == 2

    def test_on_epoch_end_logs(self, tmp_path):
        cb = HFTrainCardCallback()
        args = make_args(str(tmp_path))
        cb.on_train_begin(args, make_state(), make_control())
        cb.on_epoch_end(args, make_state(epoch=1.0), make_control())
        logs = cb.reporter.get_state()["logs"]
        assert any("1" in l["line"] for l in logs)


# ─── on_train_end ────────────────────────────────────────────────────

class TestTrainEnd:
    def test_marks_done(self, tmp_path):
        cb = HFTrainCardCallback()
        args = make_args(str(tmp_path))
        cb.on_train_begin(args, make_state(), make_control())
        cb.on_train_end(args, make_state(), make_control())
        assert cb.reporter.get_state()["phase"] == "done"


# ─── on_step_end / heartbeat ─────────────────────────────────────────

class TestHeartbeat:
    def test_heartbeat_called_on_step(self, tmp_path):
        cb = HFTrainCardCallback()
        args = make_args(str(tmp_path))
        cb.on_train_begin(args, make_state(), make_control())
        t_before = cb.reporter.get_state()["last_heartbeat"]
        time.sleep(0.05)
        cb.on_step_end(args, make_state(step=1), make_control())
        t_after = cb.reporter.get_state()["last_heartbeat"]
        assert t_after >= t_before


# ─── System stats (mocked) ───────────────────────────────────────────

class TestSystemStats:
    def test_sample_system_populates_stats(self, tmp_path):
        with patch("psutil.cpu_percent", return_value=55.0), \
             patch("psutil.virtual_memory") as mock_vm, \
             patch("psutil.disk_io_counters") as mock_disk:
            mock_vm.return_value = SimpleNamespace(used=16 * 1024 ** 3, total=32 * 1024 ** 3)
            mock_disk.return_value = SimpleNamespace(read_bytes=500 * 1024 ** 2, write_bytes=200 * 1024 ** 2)

            cb = HFTrainCardCallback()
            args = make_args(str(tmp_path))
            cb.on_train_begin(args, make_state(), make_control())
            cb._sample_system()

            stats = cb.reporter.get_state()["system"]
            assert stats["cpu_percent"] == 55.0
            assert stats["ram_used_gb"] == 16.0
            assert stats["ram_total_gb"] == 32.0

    def test_sample_system_graceful_on_import_error(self, tmp_path):
        """Should not raise even if psutil/pynvml not installed."""
        with patch.dict("sys.modules", {"psutil": None, "pynvml": None}):
            cb = HFTrainCardCallback()
            args = make_args(str(tmp_path))
            cb.on_train_begin(args, make_state(), make_control())
            # Should not raise
            cb._sample_system()
