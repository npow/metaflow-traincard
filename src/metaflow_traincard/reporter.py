"""
Core Reporter SDK — framework-agnostic training event ingestion.

Thread-safe, async-flush, crash-safe writes. Works locally and on
any remote Metaflow compute (Batch, K8s) without external dependencies.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_STALL_TIMEOUT_DEFAULT = 300  # seconds before flagging GPU stall
_MAX_LOG_LINES = 500
_MAX_METRIC_POINTS = 100_000


class Reporter:
    """
    Framework-agnostic training event collector.

    Usage (raw PyTorch):
        reporter = Reporter()
        for step, batch in enumerate(loader):
            loss = train_step(batch)
            reporter.metric("loss", loss, step)
        reporter.finish()

    All methods are thread-safe.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        flush_interval: float = 5.0,
        flush_every_n_steps: int = 50,
        stall_timeout: float = _STALL_TIMEOUT_DEFAULT,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Args:
            output_dir:         Directory for events.jsonl + latest.json.
                                Defaults to /tmp/traincard/<pid>.
            flush_interval:     Seconds between background flushes.
            flush_every_n_steps: Also flush after this many metric() calls.
            stall_timeout:      Seconds of no-progress before stall warning.
            rank:               Distributed rank (0 = main reporter).
            world_size:         Total distributed processes.
        """
        if output_dir is None:
            output_dir = f"/tmp/traincard/{os.getpid()}"
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._flush_interval = flush_interval
        self._flush_every_n_steps = flush_every_n_steps
        self._stall_timeout = stall_timeout
        self._rank = rank
        self._world_size = world_size
        self._step_since_flush = 0

        self._state: Dict[str, Any] = {
            "phase": "init",
            "step": 0,
            "epoch": 0,
            "start_time": time.time(),
            "last_update_time": time.time(),
            "last_heartbeat": time.time(),
            "rank": rank,
            "world_size": world_size,
            "metrics": {},      # name -> [{step, value}, ...]
            "system": {},       # latest system snapshot
            "checkpoints": [],  # [{path, step, time, metadata}]
            "logs": [],         # [{time, line, level}]
            "failure": None,    # {type, message, traceback} if crashed
            "stalled": False,
            "restart_count": 0,
        }

        # Load existing state for resume continuity
        self._latest_path = self._dir / "latest.json"
        self._events_path = self._dir / "events.jsonl"
        self._checkpoints_path = self._dir / "checkpoints.json"
        self._resume_if_exists()

        self._lock = threading.Lock()
        self._closed = False

        # Register SIGTERM handler for crash-safe flush
        self._prev_sigterm = signal.signal(signal.SIGTERM, self._on_sigterm)

        # Background flush thread
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="traincard-flush"
        )
        self._flush_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a scalar training metric."""
        if self._rank != 0:
            return  # Only main rank writes
        with self._lock:
            step = step if step is not None else self._state["step"]
            self._state["step"] = max(self._state["step"], step)
            self._state["last_update_time"] = time.time()
            self._state["stalled"] = False

            bucket = self._state["metrics"].setdefault(name, [])
            # Compact: skip if identical step already recorded
            if not bucket or bucket[-1]["step"] != step:
                bucket.append({"step": step, "value": float(value)})
                # Cap to prevent unbounded growth
                if len(bucket) > _MAX_METRIC_POINTS:
                    bucket[:] = bucket[-_MAX_METRIC_POINTS:]

        self._append_event("metric", {
            "name": name,
            "value": float(value),
            "step": step,
            "tags": tags or {},
        })
        self._step_since_flush += 1
        if self._step_since_flush >= self._flush_every_n_steps:
            self._flush()
            self._step_since_flush = 0

    def log(self, line: str, level: str = "info") -> None:
        """Append a structured log line."""
        with self._lock:
            self._state["logs"].append({
                "time": time.time(),
                "line": str(line),
                "level": level,
            })
            # Keep only most recent lines
            if len(self._state["logs"]) > _MAX_LOG_LINES:
                self._state["logs"] = self._state["logs"][-_MAX_LOG_LINES:]

    def phase(self, phase_name: str) -> None:
        """Mark the current training phase: 'train', 'eval', 'save', 'done'."""
        with self._lock:
            self._state["phase"] = phase_name
        self._append_event("phase", {"phase": phase_name})

    def checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a saved checkpoint."""
        with self._lock:
            info = {
                "path": str(path),
                "step": self._state["step"],
                "time": time.time(),
                "metadata": metadata or {},
            }
            self._state["checkpoints"].append(info)
        self._append_event("checkpoint", info)
        # Sync checkpoints file for the card to read
        self._flush_checkpoints()

    def system(self, stats: Dict[str, Any]) -> None:
        """Record system telemetry snapshot (GPU, CPU, RAM, etc.)."""
        with self._lock:
            self._state["system"] = stats
            self._state["last_update_time"] = time.time()
        self._append_event("system", stats)

    def heartbeat(self) -> None:
        """Signal liveness — resets stall detection timer."""
        with self._lock:
            self._state["last_heartbeat"] = time.time()
            self._state["stalled"] = False

    def failure(
        self,
        exc_type: str,
        message: str,
        traceback: Optional[str] = None,
    ) -> None:
        """Record a training failure for the failure-summary card section."""
        with self._lock:
            self._state["failure"] = {
                "type": exc_type,
                "message": message,
                "traceback": traceback,
                "step": self._state["step"],
                "time": time.time(),
                "oom_suspected": "out of memory" in message.lower(),
            }
        self._append_event("failure", self._state["failure"])
        self._flush()

    def epoch(self, epoch_num: int) -> None:
        """Update the current epoch counter."""
        with self._lock:
            self._state["epoch"] = epoch_num

    def finish(self) -> None:
        """Flush all pending state and stop the background thread."""
        self._closed = True
        with self._lock:
            self._state["phase"] = "done"
        self._flush()
        # Restore original SIGTERM handler
        try:
            signal.signal(signal.SIGTERM, self._prev_sigterm)
        except Exception:
            pass

    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot of the current state (safe to serialize)."""
        with self._lock:
            import copy
            return copy.deepcopy(self._state)

    @property
    def output_dir(self) -> Path:
        return self._dir

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resume_if_exists(self) -> None:
        """Restore metric history from a prior run for resume continuity."""
        if self._latest_path.exists():
            try:
                prior = json.loads(self._latest_path.read_text())
                # Only carry over metric history for visual continuity
                for name, points in prior.get("metrics", {}).items():
                    self._state["metrics"][name] = points
                self._state["restart_count"] = prior.get("restart_count", 0) + 1
                self._state["step"] = prior.get("step", 0)
                # Insert a visual discontinuity marker
                for points in self._state["metrics"].values():
                    points.append({"step": self._state["step"], "value": None, "_restart": True})
            except Exception:
                pass

    def _append_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Append a single event to events.jsonl (best-effort, non-blocking)."""
        event = {"type": event_type, "ts": time.time(), **data}
        try:
            with open(self._events_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

    def _flush(self) -> None:
        """Atomically write latest.json from current state."""
        with self._lock:
            # Check for stall
            elapsed_since_update = time.time() - self._state["last_update_time"]
            if (
                elapsed_since_update > self._stall_timeout
                and self._state["phase"] not in ("done", "init")
            ):
                self._state["stalled"] = True
            snapshot = {
                **self._state,
                "elapsed_seconds": time.time() - self._state["start_time"],
            }
        # Atomic write via tmp file
        tmp = self._latest_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(snapshot, indent=2))
            tmp.replace(self._latest_path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _flush_checkpoints(self) -> None:
        """Write checkpoints.json (separate file for quick card reads)."""
        with self._lock:
            checkpoints = list(self._state["checkpoints"])
        tmp = self._checkpoints_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(checkpoints, indent=2))
            tmp.replace(self._checkpoints_path)
        except Exception:
            pass

    def _flush_loop(self) -> None:
        """Background thread: flush every flush_interval seconds."""
        while not self._closed:
            time.sleep(self._flush_interval)
            try:
                self._flush()
            except Exception:
                pass

    def _on_sigterm(self, signum, frame) -> None:
        """SIGTERM handler: flush state before dying."""
        try:
            self._flush()
        except Exception:
            pass
        # Re-raise with the original handler
        try:
            if callable(self._prev_sigterm):
                self._prev_sigterm(signum, frame)
        except Exception:
            pass
