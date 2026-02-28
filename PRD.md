# 📄 Product Requirements Document

## Product: Metaflow TrainCard (working name)

## 1. Vision

Turn Metaflow into the best place to run and monitor long-running model training jobs.

Today:

* Users stare at raw logs.
* GPU runs are opaque.
* Metrics are fragmented (Trainer logs, WandB, CloudWatch, etc.).
* Debugging mid-run is painful.
* Dynamic cards exist, but no canonical training card.

Goal:
Create a **real-time training observability system inside Metaflow Cards** that works across frameworks and environments.

This becomes:

> The default monitoring surface for LLM fine-tuning in Metaflow.

---

# 2. Problem Statement

LLM fine-tuning is the fastest growing Metaflow workload (LLaMA, LoRA, PEFT, Dolly, etc.). Pain points:

1. No structured visibility during multi-hour GPU runs.
2. Raw terminal logs are noisy and unstructured.
3. GPU utilization and throughput are not visible.
4. Checkpoints are opaque.
5. Failures require log scrolling archaeology.
6. Observability solutions (W&B, MLflow) are optional, external, or overkill.
7. Dynamic Cards were designed for live updates but lack a canonical training implementation.

This creates friction exactly where compute cost is highest.

---

# 3. Product Scope

## 3.1 In Scope

* Real-time training metric visualization
* Framework-agnostic event ingestion
* HuggingFace adapter
* GPU telemetry
* Checkpoint tracking
* Failure diagnostics
* Resume-awareness
* Artifact linking
* Multi-node awareness
* Works on:

  * Local
  * AWS Batch
  * Kubernetes
  * Any remote Metaflow compute

## 3.2 Out of Scope (for v1 full release)

* Hyperparameter tuning UI
* Model comparison dashboard across runs
* Auto early stopping orchestration
* Distributed debugging console

---

# 4. System Architecture

## 4.1 Core Components

### A. Reporter SDK (framework-agnostic)

A minimal API that training code calls:

```python
reporter.metric(name, value, step, tags={})
reporter.log(line)
reporter.phase("train" | "eval" | "save")
reporter.checkpoint(path, metadata={})
reporter.system(stats_dict)
reporter.heartbeat()
```

Responsibilities:

* Append events to JSONL
* Maintain compacted `latest_state.json`
* Flush periodically to datastore
* Handle resume continuity

No hard dependency on HuggingFace.

---

### B. Framework Adapters

Adapters wrap common frameworks:

* HuggingFace TrainerCallback
* PyTorch raw loop helper
* Lightning callback
* Accelerate wrapper
* TRL/PEFT compatible

Adapters simply translate framework events → Reporter calls.

---

### C. Storage Layer

Must support live updates inside a Metaflow step.

Design:

Each task writes to:

```
<run>/<step>/<task>/traincard/
    events.jsonl
    latest.json
    checkpoints.json
```

Written to:

* S3 / GCS / Azure
* Or local datastore depending on backend

Flush cadence:

* Every N seconds
* Or every M steps

Card polls these files.

---

### D. Dynamic Card UI

A reactive frontend embedded in Metaflow Cards.

Refreshes every 3–5 seconds.

---

# 5. User Experience

## 5.1 Card Layout

### Section 1 — Run Status

* Current phase
* Current epoch/step
* Time elapsed
* Estimated time remaining
* Global batch size
* Distributed world size

---

### Section 2 — Training Metrics (live charts)

* Loss curve
* Eval metric curve
* Learning rate
* Gradient norm
* Tokens/sec
* Samples/sec

Multi-line support for:

* Per-GPU
* Train vs Eval
* Multiple metrics

---

### Section 3 — System Telemetry

* GPU utilization
* VRAM usage
* GPU temperature
* CPU usage
* RAM usage
* Disk throughput

Support:

* Multi-GPU display
* Multi-node aggregation

---

### Section 4 — Checkpoints

* List of saved checkpoints
* Step number
* Size
* Time
* Download link (artifact)
* “Best checkpoint” indicator

---

### Section 5 — Logs (Structured)

* Tail of recent logs
* Highlight errors
* Collapsible tracebacks
* Search

---

### Section 6 — Failure Summary (if crash)

On failure:

* Show last metrics
* Show last system state
* Show exception type
* Show stack trace snippet
* Highlight likely OOM if detected

---

# 6. Advanced Capabilities

## 6.1 Resume Awareness

If run resumes:

* Continue metric timeline
* Mark discontinuity visually
* Track restart count

---

## 6.2 Multi-Node Awareness

If using DDP / FSDP:

* Aggregate metrics
* Show per-rank variance if needed
* Only allow rank-0 writes for main metrics

---

## 6.3 Distributed Failure Detection

If one worker dies:

* Show which rank
* Show last heartbeat
* Mark training degraded

---

## 6.4 GPU Stall Detection

If:

* No metric update for X minutes
* GPU utilization < threshold
* No progress in step

Card shows:

> "Training appears stalled"

---

# 7. Integration Model

## 7.1 Minimal User Code (HF)

```python
from metaflow_traincard import HFTrainCardCallback

trainer = Trainer(
    ...,
    callbacks=[HFTrainCardCallback()]
)
trainer.train()
```

## 7.2 Raw PyTorch

```python
from metaflow_traincard import Reporter

reporter = Reporter()

for step, batch in enumerate(loader):
    loss = train_step(batch)
    reporter.metric("loss", loss, step)
```

---

# 8. Non-Functional Requirements

### Performance

* <1% overhead
* Async writes
* Buffered I/O
* No blocking on object store

### Reliability

* Crash-safe writes
* Atomic state compaction
* Survive SIGTERM

### Security

* No open ports required
* Use Metaflow datastore credentials

### Scalability

* Handle 100k+ steps
* Automatic event compaction

---

# 9. Competitive Positioning

Compared to:

* WandB
* MLflow
* TensorBoard
* ClearML

TrainCard is:

* Zero external dependency
* Native to Metaflow
* Zero infra setup
* Works in Batch/K8s
* No separate dashboard
* Versioned with the run

This makes Metaflow more vertically integrated for AI workloads.

---

# 10. Why This Matters Strategically

This extension:

1. Strengthens Metaflow’s position in LLM fine-tuning.
2. Reduces need for external tracking systems.
3. Makes GPU runs feel first-class.
4. Turns Cards into a real observability surface.
5. Enables future features:

   * Training comparisons
   * Auto early stopping
   * Distributed introspection
   * Agentic model self-monitoring

This is not “a callback.”
It’s training observability infrastructure.

---

# 11. Naming Implications

If this becomes foundational, avoid:

* HF-only naming
* “LLM”-specific naming

This should evolve into:

* EvalCard
* InferenceCard
* AgentLoopCard
* RewardModelCard

So think platform, not plugin.
