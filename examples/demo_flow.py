"""
TrainCard Demo Flow

Simulates a 3-epoch LLM fine-tuning run with realistic metrics, system
telemetry, checkpoints, and a mid-run eval phase.  Produces a @card
viewable in the Metaflow UI or with `metaflow card view`.

Run:
    python examples/demo_flow.py run

Then view the card:
    metaflow card view TrainCardDemoFlow/<run_id>/train
"""

import math
import random
import sys
import time
import os

# Ensure the src package is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metaflow import FlowSpec, card, step, current
from metaflow_traincard import Reporter


class TrainCardDemoFlow(FlowSpec):
    """
    Demonstrates TrainCard observability for a simulated LLM fine-tuning run.

    Architecture:
        start → train → end

    The `train` step simulates 3 epochs × 30 steps of gradient descent with:
      - Exponential loss decay with noise
      - Cosine LR schedule
      - Periodic evaluation
      - Checkpoint saving after each epoch
      - Realistic GPU/CPU telemetry
    """

    @step
    def start(self):
        print("Starting TrainCard demo flow.")
        self.next(self.train)

    @card(type="traincard")
    @step
    def train(self):
        """Simulated multi-epoch training step with TrainCard instrumentation."""
        rng = random.Random(42)

        reporter = Reporter(
            output_dir=f"/tmp/traincard_demo_{os.getpid()}",
            flush_interval=2,
        )

        num_epochs = 3
        steps_per_epoch = 30
        total_steps = num_epochs * steps_per_epoch

        reporter.log(f"Demo run: {num_epochs} epochs × {steps_per_epoch} steps/epoch")
        reporter.log(f"Model: tiny-llm-demo  Batch size: 16  LR: 2e-4")

        for epoch in range(num_epochs):
            reporter.epoch(epoch + 1)
            reporter.phase("train")
            reporter.log(f"── Epoch {epoch + 1}/{num_epochs} ──")

            for local_step in range(steps_per_epoch):
                global_step = epoch * steps_per_epoch + local_step

                # Realistic loss curve: exponential decay + noise + occasional spikes
                base_loss = 2.5 * math.exp(-0.04 * global_step)
                noise = rng.gauss(0, 0.08) * math.exp(-0.02 * global_step)
                spike = 0.4 if local_step == 15 and epoch == 0 else 0  # Learning rate spike
                loss = max(0.05, base_loss + noise + spike)

                # Cosine LR schedule
                lr = 2e-4 * 0.5 * (1 + math.cos(math.pi * global_step / total_steps))

                # Gradient norm (increases near spikes)
                grad_norm = rng.gauss(1.2, 0.3) + (0.8 if spike else 0)

                # Throughput metrics
                tokens_per_sec = rng.gauss(12_500, 800)
                samples_per_sec = tokens_per_sec / 512  # 512 tok/sample

                reporter.metric("train/loss", loss, global_step)
                reporter.metric("train/learning_rate", lr, global_step)
                reporter.metric("train/grad_norm", abs(grad_norm), global_step)
                reporter.metric("train/tokens_per_sec", tokens_per_sec, global_step)
                reporter.metric("train/samples_per_sec", samples_per_sec, global_step)

                # GPU telemetry (simulated)
                reporter.system({
                    "gpu_utilization": [
                        rng.gauss(88, 4),
                        rng.gauss(86, 5),
                    ],
                    "gpu_memory_used_gb": [
                        round(rng.gauss(18.5, 0.3), 2),
                        round(rng.gauss(18.2, 0.3), 2),
                    ],
                    "gpu_memory_total_gb": [24.0, 24.0],
                    "gpu_temperature": [
                        rng.gauss(74, 2),
                        rng.gauss(72, 2),
                    ],
                    "cpu_percent": rng.gauss(35, 8),
                    "ram_used_gb": round(rng.gauss(42, 2), 1),
                    "ram_total_gb": 64.0,
                    "disk_read_mbps": rng.gauss(320, 40),
                    "disk_write_mbps": rng.gauss(180, 25),
                })

                reporter.heartbeat()

                if global_step % 10 == 0:
                    reporter.log(
                        f"[E{epoch+1} S{local_step+1:2d}] "
                        f"loss={loss:.4f}  lr={lr:.2e}  "
                        f"grad_norm={abs(grad_norm):.3f}  "
                        f"tok/s={tokens_per_sec:.0f}"
                    )

                time.sleep(0.03)  # Simulate GPU compute time

            # ── End-of-epoch evaluation ──────────────────────────────────
            reporter.phase("eval")
            reporter.log(f"Running validation — epoch {epoch + 1}")
            eval_loss = loss * rng.gauss(0.93, 0.02)  # Eval slightly better
            eval_ppl = math.exp(eval_loss)
            reporter.metric("eval/loss", eval_loss, global_step)
            reporter.metric("eval/perplexity", eval_ppl, global_step)
            reporter.log(f"Eval: loss={eval_loss:.4f}  ppl={eval_ppl:.2f}")
            time.sleep(0.2)

            # ── Checkpoint ───────────────────────────────────────────────
            reporter.phase("save")
            ckpt_path = f"/tmp/traincard_demo_ckpt/epoch-{epoch+1}-step-{global_step}"
            reporter.checkpoint(
                ckpt_path,
                metadata={
                    "epoch": epoch + 1,
                    "step": global_step,
                    "eval_loss": round(eval_loss, 4),
                    "eval_perplexity": round(eval_ppl, 2),
                },
            )
            reporter.log(f"Checkpoint saved → {ckpt_path}")
            time.sleep(0.05)

        reporter.phase("done")
        reporter.finish()
        reporter.log("Training complete!")

        # Store state in artifact for the card to render
        self.traincard_state = reporter.get_state()
        self.next(self.end)

    @step
    def end(self):
        state = self.traincard_state
        final_loss = state["metrics"].get("train/loss", [{}])[-1].get("value", "?")
        eval_loss = state["metrics"].get("eval/loss", [{}])[-1].get("value", "?")
        print(f"\n{'─'*50}")
        print(f"  TrainCard Demo Complete")
        print(f"  Final train loss : {final_loss:.4f}" if isinstance(final_loss, float) else "  Final train loss : ?")
        print(f"  Final eval loss  : {eval_loss:.4f}" if isinstance(eval_loss, float) else "  Final eval loss  : ?")
        print(f"  Checkpoints saved: {len(state.get('checkpoints', []))}")
        print(f"{'─'*50}")
        print("\nView the card:")
        print("  metaflow card view TrainCardDemoFlow --id latest --step train")


if __name__ == "__main__":
    TrainCardDemoFlow()
