from __future__ import annotations

from importlib import import_module

__all__ = ["FinanceTutorialPipeline", "FinanceTutorialEvaluator", "FinDERExperimentRunner"]


def __getattr__(name: str):
    if name == "FinanceTutorialPipeline":
        return import_module(".framework.pipeline", __name__).FinanceTutorialPipeline
    if name == "FinanceTutorialEvaluator":
        return import_module(".framework.evaluation", __name__).FinanceTutorialEvaluator
    if name == "FinDERExperimentRunner":
        return import_module(".framework.finder_experiment", __name__).FinDERExperimentRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
