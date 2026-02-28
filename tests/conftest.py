"""Shared fixtures for TrainCard tests."""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from metaflow_traincard.reporter import Reporter


@pytest.fixture
def tmp_dir(tmp_path):
    """A clean temp directory for each test."""
    return tmp_path


@pytest.fixture
def reporter(tmp_dir):
    """A fresh Reporter writing to a temp directory."""
    r = Reporter(
        output_dir=str(tmp_dir / "traincard"),
        flush_interval=60,   # Don't auto-flush during tests
        flush_every_n_steps=9999,
        stall_timeout=9999,
    )
    yield r
    r.finish()


@pytest.fixture
def minimal_state():
    """A minimal valid state dict."""
    return {
        "phase": "train",
        "step": 100,
        "epoch": 2,
        "start_time": time.time() - 300,
        "elapsed_seconds": 300,
        "last_update_time": time.time(),
        "last_heartbeat": time.time(),
        "rank": 0,
        "world_size": 1,
        "metrics": {
            "loss": [{"step": i * 10, "value": 2.0 / (1 + i * 0.1)} for i in range(11)],
            "learning_rate": [{"step": i * 10, "value": 1e-4 * (0.95 ** i)} for i in range(11)],
        },
        "system": {
            "gpu_utilization": [85.0, 83.0],
            "gpu_memory_used_gb": [12.5, 12.3],
            "gpu_memory_total_gb": [24.0, 24.0],
            "gpu_temperature": [72.0, 70.0],
            "cpu_percent": 42.0,
            "ram_used_gb": 28.4,
            "ram_total_gb": 64.0,
        },
        "checkpoints": [
            {
                "path": "/tmp/checkpoint-step-50",
                "step": 50,
                "time": time.time() - 150,
                "metadata": {"epoch": 1},
            },
            {
                "path": "/tmp/checkpoint-step-100",
                "step": 100,
                "time": time.time() - 10,
                "metadata": {"epoch": 2, "eval_loss": 1.45},
            },
        ],
        "logs": [
            {"time": time.time() - 60, "line": "[step 80] loss=1.523", "level": "info"},
            {"time": time.time() - 30, "line": "[step 90] loss=1.421", "level": "info"},
            {"time": time.time() - 10, "line": "[step 100] loss=1.312", "level": "info"},
        ],
        "failure": None,
        "stalled": False,
        "restart_count": 0,
    }


@pytest.fixture
def failure_state(minimal_state):
    """A state dict representing a crashed run."""
    s = dict(minimal_state)
    s["phase"] = "train"
    s["failure"] = {
        "type": "RuntimeError",
        "message": "CUDA out of memory. Tried to allocate 2.00 GiB",
        "traceback": "Traceback (most recent call last):\n  File 'train.py', line 42\n    loss.backward()\nRuntimeError: CUDA out of memory.",
        "step": 100,
        "time": time.time(),
        "oom_suspected": True,
    }
    return s
