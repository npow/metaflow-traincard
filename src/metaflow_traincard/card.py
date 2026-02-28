"""
TrainCard — Metaflow custom card type.

Registered via entry point:
    metaflow.cards = traincard = metaflow_traincard.card:TrainCard

The card reads the `traincard_state` artifact from the Metaflow task
and renders a full-page HTML observability dashboard.

Live-update pattern (from inside the step):
    reporter = Reporter()
    reporter.card_updater = current.card   # attach for live updates
"""

from __future__ import annotations

from typing import Any, Dict

from ._html import render_card_html

try:
    from metaflow.cards import MetaflowCard
    _HAS_METAFLOW = True
except ImportError:
    # Graceful degradation when Metaflow is not installed (e.g., CI)
    MetaflowCard = object
    _HAS_METAFLOW = False


_ARTIFACT_NAME = "traincard_state"


class TrainCard(MetaflowCard):
    """
    Metaflow card type 'traincard'.

    Renders a real-time training observability dashboard from the
    ``traincard_state`` artifact produced by :class:`~metaflow_traincard.Reporter`.

    Usage::

        from metaflow import FlowSpec, step, card
        from metaflow_traincard import Reporter

        class MyFlow(FlowSpec):
            @card(type='traincard')
            @step
            def train(self):
                reporter = Reporter()
                for step, batch in enumerate(loader):
                    loss = train_step(batch)
                    reporter.metric("loss", loss, step)
                reporter.finish()
                self.traincard_state = reporter.get_state()  # ← required
                self.next(self.end)
    """

    type = "traincard"
    ALLOW_USER_COMPONENTS = False

    def render(self, task) -> str:
        """Called by the Metaflow UI to produce the card HTML."""
        state = self._read_state(task)
        return render_card_html(state)

    # Metaflow >= 2.11 calls render_runtime for live refreshes
    def render_runtime(self, task, data) -> str:
        state = self._read_state(task)
        return render_card_html(state)

    # ------------------------------------------------------------------

    @staticmethod
    def _read_state(task) -> Dict[str, Any]:
        try:
            return task[_ARTIFACT_NAME].data
        except Exception:
            pass
        # Fallback: try older dict-style access
        try:
            return task.data[_ARTIFACT_NAME]
        except Exception:
            pass
        return {}


# ------------------------------------------------------------------
# Standalone render helper (useful for testing / CI)
# ------------------------------------------------------------------

def render_state(state: Dict[str, Any]) -> str:
    """Render card HTML from a state dict without a Metaflow task object."""
    return render_card_html(state)
