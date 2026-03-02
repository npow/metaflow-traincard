"""Framework adapters for TrainCard Reporter."""


def __getattr__(name: str):
    if name == "HFTrainCardCallback":
        from .huggingface import HFTrainCardCallback
        globals()["HFTrainCardCallback"] = HFTrainCardCallback
        return HFTrainCardCallback
    raise AttributeError(f"module 'metaflow_traincard.adapters' has no attribute {name!r}")


__all__ = ["HFTrainCardCallback"]
