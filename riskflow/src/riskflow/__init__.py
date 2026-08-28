"""riskflow — credit risk scorecards, rule mining and decision policies.

    from riskflow import Settings, train, ScoringBundle

    result = train(data=df, label="is_bad", settings=Settings())
    decisions = ScoringBundle.load(result.run.bundle_path).score(new_applications)

Training uses scikit-learn and XGBoost; scoring uses numpy and pandas alone,
because everything a run produces is exported into one JSON bundle.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Settings",
    "ScoringBundle",
    "DecisionPolicy",
    "Registry",
    "train",
    "score_batch",
    "compare_runs",
]


def __getattr__(name: str):
    # Imported lazily so `import riskflow` stays cheap and so the scoring path
    # never pulls in the training-only dependencies.
    if name == "Settings":
        from .settings import Settings

        return Settings
    if name == "ScoringBundle":
        from .bundle import ScoringBundle

        return ScoringBundle
    if name == "DecisionPolicy":
        from .policy.decision import DecisionPolicy

        return DecisionPolicy
    if name == "Registry":
        from .registry import Registry

        return Registry
    if name == "train":
        from .train import train

        return train
    if name == "score_batch":
        from .scoring import score_batch

        return score_batch
    if name == "compare_runs":
        from .compare import compare_runs

        return compare_runs
    raise AttributeError(f"module 'riskflow' has no attribute '{name}'")
