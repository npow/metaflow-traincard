"""
metaflow-traincard — Real-time training observability for Metaflow.

Quick start::

    from metaflow_traincard import Reporter

    reporter = Reporter()
    for step, batch in enumerate(loader):
        loss = train_step(batch)
        reporter.metric("loss", loss, step)
    reporter.finish()

HuggingFace::

    from metaflow_traincard import HFTrainCardCallback

    trainer = Trainer(..., callbacks=[HFTrainCardCallback()])
    trainer.train()
"""

from .reporter import Reporter
from .card import TrainCard, render_state
from .adapters.huggingface import HFTrainCardCallback

__version__ = "0.1.0"
__all__ = [
    "Reporter",
    "TrainCard",
    "render_state",
    "HFTrainCardCallback",
]
