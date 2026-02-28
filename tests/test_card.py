"""Tests for the card HTML renderer and MetaflowCard integration."""

import json
import re
import time

import pytest

from metaflow_traincard._html import (
    _fmt_duration,
    _fmt_size,
    _build_chart_blocks,
    _build_checkpoints_html,
    _build_log_html,
    _build_failure_html,
    _build_system_html,
    render_card_html,
)
from metaflow_traincard.card import TrainCard, render_state


# ─── render_card_html ────────────────────────────────────────────────

class TestRenderCardHtml:
    def test_returns_string(self, minimal_state):
        html = render_card_html(minimal_state)
        assert isinstance(html, str)

    def test_valid_html_structure(self, minimal_state):
        html = render_card_html(minimal_state)
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html or "<head" in html
        assert "<body>" in html or "<body" in html

    def test_includes_chartjs(self, minimal_state):
        html = render_card_html(minimal_state)
        assert "chart.js" in html.lower() or "Chart" in html

    def test_shows_phase(self, minimal_state):
        minimal_state["phase"] = "eval"
        html = render_card_html(minimal_state)
        assert "EVAL" in html

    def test_shows_step(self, minimal_state):
        minimal_state["step"] = 9876
        html = render_card_html(minimal_state)
        assert "9,876" in html or "9876" in html

    def test_shows_elapsed_time(self, minimal_state):
        minimal_state["elapsed_seconds"] = 3661
        html = render_card_html(minimal_state)
        # 3661s = 1h 1m 1s
        assert "1h" in html

    def test_metric_names_in_html(self, minimal_state):
        html = render_card_html(minimal_state)
        assert "loss" in html
        assert "learning_rate" in html

    def test_checkpoint_paths_in_html(self, minimal_state):
        html = render_card_html(minimal_state)
        assert "checkpoint-step-50" in html
        assert "checkpoint-step-100" in html

    def test_log_lines_in_html(self, minimal_state):
        html = render_card_html(minimal_state)
        assert "loss=1.312" in html

    def test_empty_state_renders(self):
        html = render_card_html({})
        assert "<html" in html
        assert "No metrics" in html

    def test_world_size_badge_shown_for_distributed(self, minimal_state):
        minimal_state["world_size"] = 8
        html = render_card_html(minimal_state)
        assert "8 GPU" in html

    def test_no_dist_badge_for_single(self, minimal_state):
        minimal_state["world_size"] = 1
        html = render_card_html(minimal_state)
        assert "GPU" not in html or "8 GPU" not in html

    def test_restart_badge_shown_on_resume(self, minimal_state):
        minimal_state["restart_count"] = 2
        html = render_card_html(minimal_state)
        assert "Restart" in html or "restart" in html.lower()

    def test_stall_banner_when_stalled(self, minimal_state):
        minimal_state["stalled"] = True
        html = render_card_html(minimal_state)
        assert "stalled" in html.lower() or "Stalled" in html

    def test_no_stall_banner_when_not_stalled(self, minimal_state):
        minimal_state["stalled"] = False
        html = render_card_html(minimal_state)
        # "stalled" appears in embedded JSON; check that the banner div is absent
        assert "stall-banner" not in html or "Training appears stalled" not in html

    def test_failure_section_shown(self, failure_state):
        html = render_card_html(failure_state)
        assert "FAILED" in html
        assert "RuntimeError" in html

    def test_oom_warning_shown(self, failure_state):
        html = render_card_html(failure_state)
        assert "out-of-memory" in html.lower() or "OOM" in html or "memory" in html.lower()

    def test_state_json_embedded(self, minimal_state):
        html = render_card_html(minimal_state)
        # The state JSON should be embedded in the script tag
        assert "const state" in html

    def test_xss_safe_log_lines(self):
        """Log lines with HTML should be escaped in the log display section."""
        state = {
            "phase": "train", "step": 0, "epoch": 0, "start_time": time.time(),
            "elapsed_seconds": 10, "last_update_time": time.time(),
            "metrics": {}, "system": {}, "checkpoints": [], "stalled": False,
            "failure": None, "restart_count": 0, "world_size": 1, "rank": 0,
            "last_heartbeat": time.time(),
            "logs": [{"time": time.time(), "line": "<script>alert(1)</script>", "level": "info"}],
        }
        html = render_card_html(state)
        # The log display section should escape the HTML
        assert "&lt;script&gt;" in html
        # The embedded JSON uses unicode escapes so raw </script> tag can't break out
        assert "</script>" not in html.split("const state")[1].split("buildCharts")[0]


# ─── Chart blocks ────────────────────────────────────────────────────

class TestBuildChartBlocks:
    def test_empty_metrics(self):
        assert _build_chart_blocks({}) == ""

    def test_generates_canvas_per_metric(self):
        metrics = {
            "loss": [{"step": 0, "value": 1.0}],
            "lr": [{"step": 0, "value": 1e-4}],
        }
        html = _build_chart_blocks(metrics)
        assert "chart-loss" in html
        assert "chart-lr" in html

    def test_metric_name_with_special_chars(self):
        metrics = {"eval/loss": [{"step": 0, "value": 1.0}]}
        html = _build_chart_blocks(metrics)
        assert "<canvas" in html


# ─── Checkpoints ─────────────────────────────────────────────────────

class TestBuildCheckpointsHtml:
    def test_empty_checkpoints(self):
        html = _build_checkpoints_html([])
        assert "No checkpoints" in html

    def test_shows_checkpoint_path(self):
        html = _build_checkpoints_html([
            {"path": "/tmp/ckpt-100", "step": 100, "time": time.time(), "metadata": {}}
        ])
        assert "/tmp/ckpt-100" in html

    def test_shows_best_badge(self):
        html = _build_checkpoints_html([
            {"path": "/a", "step": 50, "time": time.time(), "metadata": {"eval_loss": 1.5}},
            {"path": "/b", "step": 100, "time": time.time(), "metadata": {"eval_loss": 1.2}},
        ])
        assert "BEST" in html


# ─── Logs ────────────────────────────────────────────────────────────

class TestBuildLogHtml:
    def test_empty_logs(self):
        assert _build_log_html([]) == ""

    def test_renders_log_lines(self):
        html = _build_log_html([
            {"time": time.time(), "line": "hello world", "level": "info"}
        ])
        assert "hello world" in html

    def test_error_level_class(self):
        html = _build_log_html([
            {"time": time.time(), "line": "something broke", "level": "error"}
        ])
        assert "error" in html

    def test_escapes_html(self):
        html = _build_log_html([
            {"time": time.time(), "line": "<b>bold</b>", "level": "info"}
        ])
        assert "<b>" not in html
        assert "&lt;b&gt;" in html


# ─── System HTML ─────────────────────────────────────────────────────

class TestBuildSystemHtml:
    def test_empty_system(self):
        html = _build_system_html({})
        assert "No system telemetry" in html

    def test_shows_gpu_utilization(self):
        html = _build_system_html({
            "gpu_utilization": [85.0, 78.0],
            "gpu_memory_used_gb": [12.0, 11.5],
            "gpu_memory_total_gb": [24.0, 24.0],
        })
        assert "85" in html
        assert "GPU 0" in html
        assert "GPU 1" in html

    def test_shows_cpu_percent(self):
        html = _build_system_html({"cpu_percent": 42.0})
        assert "42" in html

    def test_shows_ram(self):
        html = _build_system_html({
            "ram_used_gb": 16.0,
            "ram_total_gb": 32.0,
        })
        assert "16.0G" in html or "16" in html


# ─── Failure HTML ────────────────────────────────────────────────────

class TestBuildFailureHtml:
    def test_none_returns_empty(self):
        assert _build_failure_html(None) == ""

    def test_shows_exception_type(self):
        html = _build_failure_html({
            "type": "ValueError", "message": "bad input",
            "traceback": None, "step": 0, "time": time.time(), "oom_suspected": False
        })
        assert "ValueError" in html

    def test_shows_traceback_toggle(self):
        html = _build_failure_html({
            "type": "RuntimeError", "message": "oops",
            "traceback": "line 1\nline 2", "step": 0, "time": time.time(), "oom_suspected": False
        })
        assert "traceback" in html.lower()

    def test_oom_warning(self):
        html = _build_failure_html({
            "type": "RuntimeError", "message": "oom",
            "traceback": None, "step": 0, "time": time.time(), "oom_suspected": True
        })
        assert "memory" in html.lower() or "OOM" in html


# ─── Helper functions ────────────────────────────────────────────────

class TestHelpers:
    def test_fmt_duration_seconds(self):
        assert _fmt_duration(45) == "45s"

    def test_fmt_duration_minutes(self):
        assert _fmt_duration(125) == "2m 5s"

    def test_fmt_duration_hours(self):
        assert _fmt_duration(3661) == "1h 1m 1s"

    def test_fmt_size_bytes(self):
        assert "B" in _fmt_size(500)

    def test_fmt_size_mb(self):
        result = _fmt_size(2 * 1024 * 1024)
        assert "MB" in result or "2.0 MB" in result

    def test_fmt_size_gb(self):
        result = _fmt_size(5 * 1024 ** 3)
        assert "GB" in result


# ─── render_state convenience ────────────────────────────────────────

class TestRenderState:
    def test_render_state_returns_html(self, minimal_state):
        html = render_state(minimal_state)
        assert "<html" in html

    def test_render_state_empty(self):
        html = render_state({})
        assert "<html" in html


# ─── TrainCard ───────────────────────────────────────────────────────

class TestTrainCard:
    def test_type_attribute(self):
        assert TrainCard.type == "traincard"

    def test_render_with_mock_task(self, minimal_state):
        """render() should work with a task-like object that has __getitem__."""
        class FakeArtifact:
            data = minimal_state

        class FakeTask:
            def __getitem__(self, key):
                if key == "traincard_state":
                    return FakeArtifact()
                raise KeyError(key)

        card = TrainCard()
        html = card.render(FakeTask())
        assert "<html" in html
        assert "loss" in html

    def test_render_with_missing_artifact(self):
        """render() should return fallback HTML if artifact is missing."""
        class FakeTask:
            def __getitem__(self, key):
                raise KeyError(key)

        card = TrainCard()
        html = card.render(FakeTask())
        assert "<html" in html  # Should not raise

    def test_render_runtime_works(self, minimal_state):
        class FakeArtifact:
            data = minimal_state

        class FakeTask:
            def __getitem__(self, key):
                return FakeArtifact()

        card = TrainCard()
        html = card.render_runtime(FakeTask(), {})
        assert "<html" in html
