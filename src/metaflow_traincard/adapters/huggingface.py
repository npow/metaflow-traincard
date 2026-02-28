"""
HuggingFace Trainer adapter.

Translates HuggingFace ``transformers.TrainerCallback`` events into
:class:`~metaflow_traincard.Reporter` calls.

Usage::

    from metaflow_traincard import HFTrainCardCallback

    trainer = Trainer(
        ...,
        callbacks=[HFTrainCardCallback()]
    )
    trainer.train()

The callback auto-creates a :class:`~metaflow_traincard.Reporter` if
none is passed.  Pass an existing reporter to share state across
callbacks::

    reporter = Reporter()
    trainer = Trainer(callbacks=[HFTrainCardCallback(reporter=reporter)])
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from ..reporter import Reporter

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    _HAS_TRANSFORMERS = True
except ImportError:
    # Build a minimal stub so the class can be defined without transformers installed.
    # The stub is only used for type-checking / import; real usage requires transformers.
    class TrainerCallback:  # type: ignore
        pass
    TrainerControl = Any
    TrainerState = Any
    TrainingArguments = Any
    _HAS_TRANSFORMERS = False


class HFTrainCardCallback(TrainerCallback):
    """
    HuggingFace ``TrainerCallback`` that feeds training events into TrainCard.

    Args:
        reporter:   Optional existing :class:`Reporter` instance.
                    If omitted, a new one is created automatically.
        output_dir: Override reporter output directory.
                    Defaults to ``{training_args.output_dir}/_traincard``.
        system_stats_interval: Seconds between GPU/CPU telemetry samples
                    (0 = disabled, requires ``pynvml`` + ``psutil``).
    """

    def __init__(
        self,
        reporter: Optional[Reporter] = None,
        output_dir: Optional[str] = None,
        system_stats_interval: float = 10.0,
    ):
        self._reporter = reporter
        self._output_dir = output_dir
        self._system_stats_interval = system_stats_interval
        self._last_system_sample = 0.0
        self._train_start = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._train_start = time.time()
        if self._reporter is None:
            out = self._output_dir or os.path.join(args.output_dir, "_traincard")
            self._reporter = Reporter(
                output_dir=out,
                flush_interval=5,
                world_size=getattr(args, "world_size", 1),
                rank=getattr(args, "local_rank", 0) or 0,
            )
        self._reporter.phase("train")
        self._reporter.log(
            f"Training started — {getattr(args, 'num_train_epochs', '?')} epochs, "
            f"batch {getattr(args, 'per_device_train_batch_size', '?')}",
            level="info",
        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self._reporter:
            self._reporter.phase("done")
            self._reporter.finish()

    # ------------------------------------------------------------------
    # Per-step / log events
    # ------------------------------------------------------------------

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if not self._reporter or not logs:
            return
        step = state.global_step if state else 0
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self._reporter.metric(key, float(value), step=step)

        # Emit system telemetry periodically
        now = time.time()
        if self._system_stats_interval > 0 and (now - self._last_system_sample) >= self._system_stats_interval:
            self._sample_system()
            self._last_system_sample = now

        # Log the full dict as a formatted line
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items()]
        self._reporter.log(f"[step {step}] {', '.join(parts)}")

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self._reporter and state:
            self._reporter.epoch(int(state.epoch or 0))
            self._reporter.phase("train")
            self._reporter.log(f"Epoch {state.epoch:.0f} started")

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self._reporter and state:
            self._reporter.log(f"Epoch {state.epoch:.0f} ended")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if not self._reporter:
            return
        self._reporter.phase("eval")
        if metrics:
            step = state.global_step if state else 0
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._reporter.metric(key, float(value), step=step)
            self._reporter.log(f"Eval: {metrics}")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not self._reporter:
            return
        self._reporter.phase("save")
        step = state.global_step if state else 0
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        self._reporter.checkpoint(
            ckpt_dir,
            metadata={"step": step, "epoch": getattr(state, "epoch", None)},
        )
        self._reporter.log(f"Checkpoint saved → {ckpt_dir}")
        self._reporter.phase("train")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self._reporter:
            self._reporter.heartbeat()

    # ------------------------------------------------------------------
    # System telemetry
    # ------------------------------------------------------------------

    def _sample_system(self) -> None:
        stats: Dict[str, Any] = {}
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            utils, mems, used_mems, temps = [], [], [], []
            for i in range(n):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                utils.append(float(u.gpu))
                used_mems.append(round(m.used / 1024 ** 3, 2))
                mems.append(round(m.total / 1024 ** 3, 2))
                temps.append(float(t))
            stats["gpu_utilization"] = utils
            stats["gpu_memory_used_gb"] = used_mems
            stats["gpu_memory_total_gb"] = mems
            stats["gpu_temperature"] = temps
        except Exception:
            pass

        try:
            import psutil  # type: ignore
            stats["cpu_percent"] = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            stats["ram_used_gb"] = round(vm.used / 1024 ** 3, 2)
            stats["ram_total_gb"] = round(vm.total / 1024 ** 3, 2)
            disk = psutil.disk_io_counters()
            if disk:
                stats["disk_read_mbps"] = round(disk.read_bytes / 1024 ** 2, 1)
                stats["disk_write_mbps"] = round(disk.write_bytes / 1024 ** 2, 1)
        except Exception:
            pass

        if stats and self._reporter:
            self._reporter.system(stats)

    # ------------------------------------------------------------------
    # Convenience accessor
    # ------------------------------------------------------------------

    @property
    def reporter(self) -> Optional[Reporter]:
        """The underlying :class:`Reporter` instance."""
        return self._reporter
