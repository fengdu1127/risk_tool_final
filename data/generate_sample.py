"""Generate sample credit-risk data for tests and demos."""
import numpy as np
import pandas as pd


np.random.seed(20260608)
n = 8000

apply_time = pd.date_range("2025-01-01", "2026-05-15", periods=n)
age = np.random.randint(20, 65, n)
income = np.random.exponential(10000, n)
debt_ratio = np.random.beta(2, 5, n)
credit_score = np.random.normal(650, 80, n).clip(300, 850)
loan_amount = np.random.exponential(50000, n)
loan_term = np.random.choice([12, 24, 36, 48, 60], n)
num_late_payment = np.random.poisson(0.5, n)
employment_years = np.random.exponential(5, n).clip(0, 40)
num_credit_cards = np.random.poisson(2, n).clip(0, 10)
pboc_overdue_cnt_24m = np.random.poisson(0.35 + 1.2 * debt_ratio, n).clip(0, 12)
pboc_overdue_days_max_24m = np.where(
    pboc_overdue_cnt_24m > 0,
    np.random.gamma(2.0, 12.0, n) * np.sqrt(pboc_overdue_cnt_24m),
    0,
).clip(0, 180)
pboc_overdue_amount_12m = (
    pboc_overdue_cnt_24m * np.random.exponential(1200, n) * np.random.uniform(0.5, 1.8, n)
).clip(0, 80000)
pboc_query_cnt_1m = np.random.poisson(0.8 + 1.8 * debt_ratio + 0.2 * pboc_overdue_cnt_24m, n).clip(0, 30)
pboc_query_cnt_3m = (pboc_query_cnt_1m + np.random.poisson(1.8 + 2.0 * debt_ratio, n)).clip(0, 60)
pboc_query_cnt_6m = (pboc_query_cnt_3m + np.random.poisson(2.5 + 2.2 * debt_ratio, n)).clip(0, 100)
multi_platform_apply_cnt_7d = np.random.poisson(0.3 + 0.25 * pboc_query_cnt_1m, n).clip(0, 30)
multi_platform_apply_cnt_30d = (
    multi_platform_apply_cnt_7d + np.random.poisson(1.0 + 0.25 * pboc_query_cnt_3m, n)
).clip(0, 80)
active_loan_cnt = np.random.poisson(1.2 + 2.5 * debt_ratio + 0.08 * pboc_query_cnt_6m, n).clip(0, 25)
outstanding_loan_balance = (
    active_loan_cnt * np.random.exponential(12000, n) + loan_amount * np.random.uniform(0.1, 0.8, n)
).clip(0, 500000)
monthly_debt_payment = outstanding_loan_balance / np.random.choice([12, 24, 36, 48, 60], n) + loan_amount / loan_term
debt_to_income = (monthly_debt_payment / np.maximum(income, 1000)).clip(0, 8)
revolving_credit_utilization = np.random.beta(2 + active_loan_cnt / 6, 4, n).clip(0, 1)
total_liability_ratio = ((outstanding_loan_balance + loan_amount) / np.maximum(income * 12, 10000)).clip(0, 12)
income_corr = income * 0.95 + np.random.normal(0, 500, n)
unstable_feat = np.where(
    np.arange(n) < n // 2,
    np.random.normal(0, 1, n),
    np.random.normal(5, 1, n),
)
channel = np.random.choice(["online", "offline", "partner"], n, p=[0.5, 0.3, 0.2])
education = np.random.choice(["high_school", "bachelor", "master", "phd"], n, p=[0.25, 0.4, 0.25, 0.1])
city_tier = np.random.choice(["T1", "T2", "T3", "T4"], n, p=[0.15, 0.3, 0.35, 0.2])

log_odds = (
    -3.6
    + 0.03 * num_late_payment
    - 0.005 * (credit_score - 650)
    + 1.1 * debt_ratio
    - 0.00001 * income
    + 0.12 * (loan_amount / 10000)
    + 0.18 * pboc_overdue_cnt_24m
    + 0.008 * pboc_overdue_days_max_24m
    + 0.000015 * pboc_overdue_amount_12m
    + 0.035 * pboc_query_cnt_1m
    + 0.025 * multi_platform_apply_cnt_7d
    + 0.045 * active_loan_cnt
    + 0.18 * debt_to_income
    + 0.65 * revolving_credit_utilization
    + 0.08 * total_liability_ratio
    + np.where(channel == "offline", 0.4, 0)
    - np.where(education == "master", 0.3, 0)
    + np.random.normal(0, 0.3, n)
)
prob = 1 / (1 + np.exp(-log_odds))
is_overdue = (np.random.uniform(0, 1, n) < prob).astype(int)

df = pd.DataFrame({
    "apply_time": apply_time,
    "age": age,
    "income": income.round(2),
    "debt_ratio": debt_ratio.round(4),
    "credit_score": credit_score.round(1),
    "loan_amount": loan_amount.round(2),
    "loan_term": loan_term,
    "num_late_payment": num_late_payment,
    "employment_years": employment_years.round(2),
    "num_credit_cards": num_credit_cards,
    "pboc_overdue_cnt_24m": pboc_overdue_cnt_24m,
    "pboc_overdue_days_max_24m": pboc_overdue_days_max_24m.round(1),
    "pboc_overdue_amount_12m": pboc_overdue_amount_12m.round(2),
    "pboc_query_cnt_1m": pboc_query_cnt_1m,
    "pboc_query_cnt_3m": pboc_query_cnt_3m,
    "pboc_query_cnt_6m": pboc_query_cnt_6m,
    "multi_platform_apply_cnt_7d": multi_platform_apply_cnt_7d,
    "multi_platform_apply_cnt_30d": multi_platform_apply_cnt_30d,
    "active_loan_cnt": active_loan_cnt,
    "outstanding_loan_balance": outstanding_loan_balance.round(2),
    "monthly_debt_payment": monthly_debt_payment.round(2),
    "debt_to_income": debt_to_income.round(4),
    "revolving_credit_utilization": revolving_credit_utilization.round(4),
    "total_liability_ratio": total_liability_ratio.round(4),
    "income_corr": income_corr.round(2),
    "unstable_feat": unstable_feat.round(4),
    "channel": channel,
    "education": education,
    "city_tier": city_tier,
    "is_overdue": is_overdue,
})

for col in ["income", "employment_years", "num_credit_cards"]:
    mask = np.random.random(n) < 0.05
    df.loc[mask, col] = np.nan

df.to_csv("data/sample.csv", index=False)
print("sample data saved: data/sample.csv")
print(f"rows: {len(df)}, columns: {df.shape[1]}")
print(f"bad rate: {df['is_overdue'].mean():.3f}")
print(df.head())
