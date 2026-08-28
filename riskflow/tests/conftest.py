from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskflow.data.schema import infer_schema
from riskflow.data.splitting import split
from riskflow.data.synth import make_applications
from riskflow.features.woe import WoeTransformer
from riskflow.settings import Settings

LABEL = "is_bad"


@pytest.fixture(scope="session")
def applications() -> pd.DataFrame:
    return make_applications(n_rows=4000, seed=7)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings().merged(
        {
            "split": {"time_col": "apply_time", "oot_months": 3},
            "model": {"search_iterations": 4, "cv_folds": 3},
            "cutoffs": {"segment_features": ["channel"], "segment_min_rows": 100},
            "report": {"make_plots": False},
        }
    )


@pytest.fixture(scope="session")
def schema(applications):
    return infer_schema(applications, LABEL, time_col="apply_time", id_col="application_id")


@pytest.fixture(scope="session")
def parts(applications, settings):
    return split(applications, LABEL, settings.split)


@pytest.fixture(scope="session")
def woe(parts, schema, settings):
    return WoeTransformer.fit(parts.train, schema, settings.binning)


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    """A small, fully deterministic frame for unit-level assertions."""
    rng = np.random.default_rng(0)
    n = 600
    risk = rng.uniform(0, 1, n)
    return pd.DataFrame(
        {
            "score_like": np.round(risk, 4),
            "steps": rng.integers(0, 5, n),
            "grade": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
            "sparse": np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n)),
            LABEL: (rng.random(n) < risk * 0.5).astype(int),
        }
    )
