# forecastFiles/core.py
import pandas as pd
import numpy as np
from datetime import timedelta

WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

EPS = 1e-9

# ---------------- Forecast Helpers ----------------
def sma_series(ts, n):
    """Return 1-step-ahead in-sample SMA fitted series."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    return ts.rolling(window=n, min_periods=n).mean().shift(1)

def sma_flat_forecast_from_actuals(ts, n, steps=1):
    """Return flat forecast based on last n actuals."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    s = ts.dropna()
    if len(s) < n:
        return [np.nan] * steps
    val = float(s.iloc[-n:].mean())
    return [val] * steps

# ---------------- Metrics ----------------
def compute_metrics(ts, pred):
    """Exact metric computation consistent with original script."""
    df = pd.concat([ts, pred], axis=1).dropna()
    if df.empty:
        return {}

    y = df.iloc[:, 0].values.astype(float)
    yhat = df.iloc[:, 1].values.astype(float)
    N = len(y)

    rmse = np.sqrt(np.mean((y - yhat) ** 2))
    mean_actual = np.mean(y) if N > 0 else EPS
    pct_rmse = rmse / (mean_actual + EPS) * 100

    mask = np.abs(y) > 0
    pct_mape = np.mean(np.abs((y[mask] - yhat[mask]) / (y[mask] + EPS))) * 100 if mask.sum() > 0 else None
    smape = np.mean(2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + EPS)) * 100
    pct_bias = (np.sum(yhat - y) / (np.sum(np.abs(y)) + EPS)) * 100

    return {
        "RMSE": rmse,
        "pct_rmse": pct_rmse,
        "pct_mape": pct_mape,
        "smape": smape,
        "pct_bias": pct_bias
    }

def calculate_champion(ts, n_candidates):
    """
    Iterates through n_candidates, fits SMA, calculates metrics,
    and returns the best 'n' based on simple_composite score.
    """
    best_n = None
    best_score = np.inf
    best_metrics = {}
    best_series = None

    for n in n_candidates:
        if n >= len(ts.dropna()):
            continue

        pred = sma_series(ts, n)
        # Only consider if we have overlap
        if pd.concat([ts, pred], axis=1).dropna().empty:
            continue

        m = compute_metrics(ts, pred)

        # Simple composite: avg of pct_rmse, smape, pct_mape, abs(pct_bias)
        comps = [m.get("pct_rmse"), m.get("smape"), m.get("pct_mape")]
        if m.get("pct_bias") is not None:
            comps.append(abs(m.get("pct_bias")))

        valid_comps = [c for c in comps if c is not None]
        composite = np.mean(valid_comps) if valid_comps else np.inf

        if composite < best_score:
            best_score = composite
            best_n = n
            best_metrics = m
            best_metrics["method"] = f"SMA_{n}"
            best_metrics["simple_composite"] = composite
            best_series = pred

    return best_n, best_metrics, best_series

# ---------------- Holiday / Adjustment Logic ----------------
def apply_adjustment(dates, values, adjustment, holiday_set):
    """
    Applies nextWorkingDay, previousWorkingDay, or dropTransaction.
    Returns (pd.DatetimeIndex, list_of_values)
    """
    if adjustment not in ["nextWorkingDay", "previousWorkingDay", "dropTransaction"]:
        # normalize: return pd.DatetimeIndex and list for consistency
        return pd.DatetimeIndex(dates), list(values)

    new_dates, new_vals = [], []
    for d, v in zip(dates, values):
        curr = pd.to_datetime(d)
        def is_bad(x): return x.weekday() >= 5 or x.date() in holiday_set

        if is_bad(curr):
            if adjustment == "dropTransaction":
                continue  # Skip entirely

            step = 1 if adjustment == "nextWorkingDay" else -1
            # move until not bad (guard to avoid infinite loops)
            attempts = 0
            while is_bad(curr) and attempts < 30:
                curr += timedelta(days=step)
                attempts += 1

            # if still bad after attempts, skip
            if is_bad(curr):
                continue

        new_dates.append(pd.to_datetime(curr))
        new_vals.append(v)

    return pd.DatetimeIndex(new_dates), new_vals

# ---------------- Outlier filter ----------------
def apply_two_sigma_filter(df, sigma=2.0):
    """
    Keep rows whose abs(amount) lies within mean ± sigma*std.
    Returns filtered copy of df.
    """
    df = df.copy()
    s = df["amount"].abs()
    mean = s.mean()
    std = s.std(ddof=0)  # population std; change ddof if you prefer sample std
    if std == 0 or np.isnan(std):
        return df.copy()
    low, high = mean - sigma * std, mean + sigma * std
    mask = (s >= low) & (s <= high)
    return df[mask].copy()
