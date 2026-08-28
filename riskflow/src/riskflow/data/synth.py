"""Synthetic loan-application data, for demos and tests.

The generator deliberately plants the structures the toolkit is supposed to
find: monotone risk drivers, a couple of pure noise columns, missing values, a
small high-risk pocket that only a conjunction of conditions isolates, and a
mild population shift over time so drift checks have something to see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_applications(
    n_rows: int = 12_000,
    bad_rate: float = 0.08,
    seed: int = 20240501,
    months: int = 18,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Applications arrive uniformly over `months`, with volume drifting upward.
    day_offsets = rng.integers(0, months * 30, size=n_rows)
    apply_time = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.sort(day_offsets), unit="D")
    tenure_frac = np.linspace(0.0, 1.0, n_rows)

    age = np.clip(rng.normal(36, 10, n_rows), 18, 75).round().astype(int)
    income = np.round(np.exp(rng.normal(9.6, 0.55, n_rows)), 2)
    # Later cohorts are slightly more leveraged: a real, mild covariate shift.
    debt_ratio = np.clip(rng.beta(2.2, 5.0, n_rows) + 0.05 * tenure_frac, 0.01, 0.99).round(4)
    credit_history_months = np.clip(rng.gamma(4.0, 14.0, n_rows), 0, 400).round().astype(int)
    inquiries_6m = rng.poisson(1.4 + 0.9 * tenure_frac, n_rows)
    open_accounts = rng.poisson(3.5, n_rows)
    max_delinquency = np.where(rng.random(n_rows) < 0.72, 0, rng.integers(1, 6, n_rows))
    utilization = np.clip(rng.beta(2.0, 3.0, n_rows), 0, 1).round(4)
    loan_amount = np.round(income * rng.uniform(0.2, 2.5, n_rows), -2)
    loan_term = rng.choice([6, 12, 24, 36], n_rows, p=[0.15, 0.4, 0.3, 0.15])

    channel = rng.choice(
        ["app", "web", "agent", "partner"], n_rows, p=[0.42, 0.28, 0.18, 0.12]
    )
    city_tier = rng.choice(["T1", "T2", "T3", "T4"], n_rows, p=[0.22, 0.33, 0.30, 0.15])
    employment = rng.choice(
        ["salaried", "self_employed", "contract", "unknown"], n_rows, p=[0.55, 0.25, 0.14, 0.06]
    )

    # Pure noise: the screening stage should drop these.
    noise_score = rng.normal(0, 1, n_rows).round(4)
    noise_flag = rng.integers(0, 2, n_rows)

    z = (
        -2.15
        + 2.60 * debt_ratio
        + 1.45 * utilization
        + 0.30 * inquiries_6m
        + 0.34 * max_delinquency
        - 0.021 * (age - 36)
        - 0.0045 * credit_history_months
        - 0.55 * (np.log(income) - 9.6)
        + 0.06 * open_accounts
    )
    z += pd.Series(channel).map(
        {"app": -0.10, "web": 0.0, "agent": 0.42, "partner": 0.20}
    ).to_numpy()
    z += pd.Series(city_tier).map(
        {"T1": -0.28, "T2": -0.05, "T3": 0.16, "T4": 0.34}
    ).to_numpy()
    z += pd.Series(employment).map(
        {"salaried": -0.15, "self_employed": 0.22, "contract": 0.18, "unknown": 0.40}
    ).to_numpy()
    # An interaction a linear scorecard cannot express, plus a tight high-risk
    # pocket that rule mining should isolate.
    z += 0.9 * (debt_ratio > 0.55) * (inquiries_6m >= 3)
    z += 2.4 * ((max_delinquency >= 2) & (utilization > 0.80))

    z -= z.mean() - np.log(bad_rate / (1 - bad_rate))
    prob = 1.0 / (1.0 + np.exp(-z))
    label = (rng.random(n_rows) < prob).astype(int)

    df = pd.DataFrame(
        {
            "application_id": [f"A{i:07d}" for i in range(n_rows)],
            "apply_time": apply_time,
            "age": age,
            "income": income,
            "debt_ratio": debt_ratio,
            "credit_history_months": credit_history_months,
            "inquiries_6m": inquiries_6m,
            "open_accounts": open_accounts,
            "max_delinquency": max_delinquency,
            "utilization": utilization,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "channel": channel,
            "city_tier": city_tier,
            "employment": employment,
            "noise_score": noise_score,
            "noise_flag": noise_flag,
            "is_bad": label,
        }
    )

    # Missing not at random: thin files lack bureau history, and the missingness
    # itself carries risk signal, which is why missing gets its own WOE bin.
    thin_file = rng.random(n_rows) < (0.06 + 0.10 * (df["credit_history_months"] < 24))
    df.loc[thin_file, "utilization"] = np.nan
    df.loc[rng.random(n_rows) < 0.04, "income"] = np.nan
    df.loc[rng.random(n_rows) < 0.03, "employment"] = None
    return df


def write_sample(path: str, **kwargs) -> pd.DataFrame:
    df = make_applications(**kwargs)
    df.to_csv(path, index=False)
    return df
