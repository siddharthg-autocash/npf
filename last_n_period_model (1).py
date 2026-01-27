# last_n_forecast_sma_wma.py
"""
Simplified last-n forecasting module focused only on:
 - Simple Moving Average (SMA)
 - Weighted Moving Average (WMA)

Behavior:
 - In-sample: compute 1-step fitted values for each candidate window.
 - Model selection: choose champion between SMA and WMA using a simple composite (same as before).
 - Future: finance standard FLAT forecast for all future periods equal to the last (weighted) average
           computed from the most recent n actuals (no recursive sliding with forecasted values).
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import calendar

EPS = 1e-9


# ---------------- Forecast helper functions ----------------

def sma_series(ts, n):
    """
    Return 1-step-ahead in-sample SMA fitted series.
    Value at time t is mean of previous n actuals (shifted by 1).
    Uses pandas rolling for vectorized computation.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    return ts.rolling(window=n, min_periods=n).mean().shift(1)


def wma_series(ts, n):
    """
    Return 1-step-ahead in-sample WMA fitted series.
    Uses linear weights 1..n (most recent gets highest weight).
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    weights = np.arange(1, n + 1, dtype=float)

    def _apply_wma(x):
        # x is a numpy array of length n (raw=True)
        return float(np.dot(x, weights) / weights.sum())

    return ts.rolling(window=n, min_periods=n).apply(_apply_wma, raw=True).shift(1)


def sma_flat_forecast_from_actuals(ts, n, steps=1):
    """
    Finance-standard flat SMA forecast: compute average of last n actuals and
    return it repeated for `steps` future periods.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    s = ts.dropna()
    if len(s) < n:
        return [np.nan] * steps
    val = float(s.iloc[-n:].mean())
    return [val] * steps


def wma_flat_forecast_from_actuals(ts, n, steps=1):
    """
    Finance-standard flat WMA forecast: compute weighted average of last n actuals
    (linear weights) and repeat for `steps`.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    s = ts.dropna()
    if len(s) < n:
        return [np.nan] * steps
    window = s.iloc[-n:].values.astype(float)
    weights = np.arange(1, n + 1, dtype=float)
    val = float(np.dot(window, weights) / weights.sum())
    return [val] * steps


# ---------------- Metric computation ----------------

def compute_metrics(ts, pred):
    """
    Compute a set of metrics between actual ts and predicted series pred.
    Returns a dict similar to the original script.
    """
    df = pd.concat([ts, pred], axis=1).dropna()
    if df.empty:
        return dict(RMSE=None, pct_rmse=None, pct_mape=None, smape=None, pct_bias=None, w_pct_rmse=None)

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

    # weighted pct RMSE (recent values weighted higher)
    w = np.linspace(1, 2, N); w = w / np.sum(w)
    w_pct_rmse = np.sqrt(np.sum(w * (y - yhat) ** 2)) / (mean_actual + EPS) * 100

    return dict(
        RMSE=rmse,
        pct_rmse=pct_rmse,
        pct_mape=pct_mape,
        smape=smape,
        pct_bias=pct_bias,
        w_pct_rmse=w_pct_rmse
    )


# ============================================================
# BIWEEKLY-ONLY HELPERS (NEW – CADENCE, NOT AGGREGATION)
# ============================================================

def _extract_biweekly_anchor_and_noise(records, pct_threshold=0.15):
    """
    Identify the last true biweekly anchor and noise amount.

    Anchor definition:
      |amount| >= pct_threshold * median(|amount|)

    Noise:
      All transactions AFTER the last anchor that do not qualify
      as anchors. These are netted into the next forecast.
    """

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    df = df.sort_values("date")

    abs_amounts = df["amount"].abs()
    median_amt = abs_amounts.median()

    if median_amt == 0 or np.isnan(median_amt):
        last_anchor = df["date"].iloc[-1]
        return last_anchor, 0.0

    threshold = pct_threshold * median_amt
    df["is_anchor"] = abs_amounts >= threshold

    anchor_rows = df[df["is_anchor"]]

    if anchor_rows.empty:
        last_anchor = df["date"].iloc[-1]
        noise_sum = 0.0
    else:
        last_anchor = anchor_rows["date"].iloc[-1]
        noise_sum = df[df["date"] > last_anchor]["amount"].sum()

    return last_anchor, float(noise_sum)


WEEKDAY_MAP = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

def _generate_biweekly_cadence(
    last_anchor_date,
    forecast_start_date,
    forecast_end_date,
    weekday
):
    dates = []

    if isinstance(weekday, str):
        weekday = WEEKDAY_MAP[weekday.lower()]

    # Step 1: anchor + 14 days
    current = last_anchor_date + timedelta(days=14)

    # Step 2: align to required weekday
    delta = (weekday - current.weekday()) % 7
    if delta == 0:
        delta = 7  # ensure forward movement
    current += timedelta(days=delta)

    # Step 3: move into forecast window
    while current < forecast_start_date:
        current += timedelta(days=14)

    while current <= forecast_end_date:
        dates.append(current)
        current += timedelta(days=14)

    return dates


# ============================================================
# MAIN ENTRY POINT (SIGNATURE UNCHANGED)
# ============================================================

def run_last_n_forecast(
    records,
    model_params,
    forecast_end_date,
    holiday_dates,
    forecast_start_date
):
    """
    Supports:
      - daily (aggregation)
      - weekly (aggregation, W-MON)
      - monthly (aggregation)
      - biweekly / fortnightly (CADENCE-BASED, NO AGGREGATION)
    """

    if not records:
        return {"status": "no_data"}

    freq = model_params.get("freq", "").lower()
    weekday = model_params.get("weekday")
    day_of_month = model_params.get("day_of_month")
    monthly_days = model_params.get("monthlyDays")
    adjustment = model_params.get("adjustmentOption")
    holiday_set = {pd.to_datetime(h).date() for h in holiday_dates or []}
    n_candidates = model_params["n_candidates"]
    if not isinstance(n_candidates, list):
        n_candidates = [n_candidates]

    forecast_end_date = pd.to_datetime(forecast_end_date)
    forecast_start_date = pd.to_datetime(forecast_start_date)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.sort_values(by="date", ascending=True)   
    df = df.dropna(subset=["date"])
    df = (
        df.groupby("date", as_index=False)["amount"]
        .sum()
    )

    # ========================================================
    # AGGREGATION PATH (DAILY / WEEKLY / MONTHLY) — UNCHANGED
    # ========================================================

    AGG_FREQ_MAP = {
        "daily": "D",
        "weekly": "W-MON",   # Monday–Sunday
        "monthly": "MS"
    }


    if freq not in ("biweekly", "fortnightly"):

        freq_code = AGG_FREQ_MAP.get(freq.lower(), "W-MON")

        ts = df.set_index("date")["amount"].astype(float)
        ts = ts.groupby(pd.Grouper(freq=freq_code)).sum()
        ts = ts.interpolate(limit_direction="both")
        non_null = ts.dropna()
        if non_null.empty:
            return {"status": "no_data", "reason": "no numeric values"}

        # --- Evaluate candidate SMAs and WMAs and pick best window for each
        preds = {}
        best_sma = best_wma = None
        best_rmse_sma = best_rmse_wma = np.inf

        for n in n_candidates:
            if n >= len(non_null):
                continue

            # SMA fitted 1-step series
            sma_pred = sma_series(ts, n)
            common_sma = pd.concat([non_null, sma_pred], axis=1).dropna()
            if len(common_sma) > 0:
                r_sma = np.sqrt(mean_squared_error(common_sma.iloc[:, 0], common_sma.iloc[:, 1]))
                if r_sma < best_rmse_sma:
                    best_sma, best_rmse_sma = n, r_sma

            # WMA fitted 1-step series
            wma_pred = wma_series(ts, n)
            common_wma = pd.concat([non_null, wma_pred], axis=1).dropna()
            if len(common_wma) > 0:
                r_wma = np.sqrt(mean_squared_error(common_wma.iloc[:, 0], common_wma.iloc[:, 1]))
                if r_wma < best_rmse_wma:
                    best_wma, best_rmse_wma = n, r_wma

        # Build preds dict only for the chosen windows (if any)
        if best_sma:
            preds[f"SMA_{best_sma}"] = sma_series(ts, best_sma)
        if best_wma:
            preds[f"WMA_{best_wma}"] = wma_series(ts, best_wma)

        if not preds:
            return {"status": "no_models", "reason": "no valid SMA/WMA models for candidate windows"}

        # --- Score models
        metrics = []
        for method, pred in preds.items():
            m = compute_metrics(ts, pred)
            m["method"] = method
            comp_vals = [
                m.get("pct_rmse"), m.get("smape"), m.get("pct_mape"),
                abs(m.get("pct_bias")) if m.get("pct_bias") is not None else None,
                m.get("w_pct_rmse")
            ]
            comp_vals = [v for v in comp_vals if v is not None]
            m["simple_composite"] = float(np.mean(comp_vals)) if comp_vals else None
            metrics.append(m)

        dfm = pd.DataFrame(metrics).set_index("method")
        if dfm.empty:
            return {"status": "no_models", "reason": "no valid models produced"}

        # Champion model selection
        champ = dfm.sort_values("simple_composite").index[0]
        champion_series = preds[champ].copy()

        future_dates = []

        # last actual index (timestamp)
        last_actual = ts.dropna().index[-1]
        # --- FIX: If monthly multi-day requested, first add leftover monthlyDays in current month (after last_actual)
        if freq.lower() == "monthly" and monthly_days:
            year = last_actual.year
            month = last_actual.month
            for entry in monthly_days:
                day = entry.get("day")
                if day is None:
                    continue
                last_day = calendar.monthrange(year, month)[1]
                day = min(day, last_day)
                dt = pd.Timestamp(year, month, day)
                if dt > last_actual and dt <= forecast_end_date:
                    future_dates.append(dt)
            # ensure sorted
            future_dates = sorted(set(future_dates))

            # Now append for subsequent months until forecast_end_date
            next_month = last_actual + pd.offsets.MonthBegin(1)
            while True:
                if next_month > forecast_end_date:
                    break
                ym_year = next_month.year
                ym_month = next_month.month
                for entry in monthly_days:
                    day = entry.get("day")
                    if day is None:
                        continue
                    last_day = calendar.monthrange(ym_year, ym_month)[1]
                    day = min(day, last_day)
                    dt = pd.Timestamp(ym_year, ym_month, day)
                    if dt <= forecast_end_date:
                        future_dates.append(dt)
                next_month = next_month + pd.offsets.MonthBegin(1)

            # final sort & unique
            future_dates = sorted(pd.DatetimeIndex(future_dates).unique())

        else:
            # Non-monthly or monthly without multi-days → use original stepping loop
            current = last_actual
            first_step = True
            while True:
                if freq.lower() == "weekly":
                    if weekday:
                        if first_step:
                            # find next matching weekday strictly after last actual
                            current = current + timedelta(days=1)
                            while current.strftime("%A") != weekday:
                                current += timedelta(days=1)
                        else:
                            current = current + timedelta(days=7)
                            while current.strftime("%A") != weekday:
                                current += timedelta(days=1)
                    else:
                        current = current + timedelta(days=7)

                elif freq.lower() == "biweekly":
                    if weekday:
                        if first_step:
                            current = current + timedelta(days=1)
                            while current.strftime("%A") != weekday:
                                current += timedelta(days=1)
                        else:
                            current = current + timedelta(days=14)
                            while current.strftime("%A") != weekday:
                                current += timedelta(days=1)
                    else:
                        current = current + timedelta(days=14)

                elif freq.lower() == "monthly":
                    # monthly single date behavior (e.g., day_of_month or default month start)
                    next_month = current + pd.offsets.MonthBegin(1)
                    # pick candidate date in next_month
                    if monthly_days:
                        # if monthlyDays present but not handled above, skip (we handled multi-day case earlier)
                        current = next_month
                    else:
                        if day_of_month:
                            last_day = calendar.monthrange(next_month.year, next_month.month)[1]
                            d = min(day_of_month, last_day)
                            current = pd.Timestamp(next_month.year, next_month.month, d)
                        else:
                            current = next_month
                else:
                    try:
                        current = current + pd.tseries.frequencies.to_offset(freq_code)
                    except Exception:
                        current = current + timedelta(days=1)

                if current > forecast_end_date:
                    break

                future_dates.append(current)
                first_step = False

        # --- Build combined forecast series (in-sample fitted + future flat forecasts)
        combined_series = champion_series.copy()

        # If needed, compute future flat forecasts per finance convention
        future_index = []
        future_forecast = []
        if len(future_dates) > 0:
            # Determine window size from champion model name
            if champ.startswith("SMA_"):
                window_n = int(champ.split("_")[1])
            elif champ.startswith("WMA_"):
                window_n = int(champ.split("_")[1])
            else:
                window_n = None

            # ---- FIX: For monthly multi-day, exclude current partial month when computing averages
            ts_for_avg = ts
            if freq.lower() == "monthly" and monthly_days:
                # exclude current month (all entries with index >= start of current month)
                # last_actual is expected to be month-start (MS), but safe: use start of that month
                first_day_of_current_month = pd.Timestamp(last_actual.year, last_actual.month, 1)
                ts_for_avg = ts[ts.index < first_day_of_current_month]

            if window_n is not None:
                future_vals = sma_flat_forecast_from_actuals(ts_for_avg, window_n, steps=len(future_dates))
            else:
                last_val = ts.dropna().iloc[-1] if len(ts.dropna()) > 0 else np.nan
                future_vals = [float(last_val)] * len(future_dates)

            # Apply percentage split for monthly frequency with monthlyDays
            if freq.lower() == "monthly" and monthly_days:
                # group dates by YYYY-MM
                grouped = {}
                for dt in future_dates:
                    key = (dt.year, dt.month)
                    grouped.setdefault(key, []).append(dt)

                new_dates = []
                new_vals = []

                # If ts_for_avg is used, base_val is same for all months (flat). If multiple months, we can use same base_val.
                # Using first element of future_vals as base_val (since flat forecast repeats same value)
                base_val = future_vals[0] if future_vals else 0.0

                # Map day -> pct for stable allocation
                day_to_pct = {int(entry.get("day")): float(entry.get("percentage", 0.0)) for entry in monthly_days}

                for (year, month), dates_in_month in grouped.items():
                    # ===== FIX START =====
                    # Check if this month has any actual payments already
                    month_start = pd.Timestamp(year, month, 1)
                    month_end = pd.Timestamp(year, month, calendar.monthrange(year, month)[1])
                    
                    # Get actual payments in this month
                    actual_in_month = ts[(ts.index >= month_start) & (ts.index <= month_end)]
                    actual_paid = actual_in_month.sum() if not actual_in_month.empty else 0.0
                    
                    # Calculate remaining amount to forecast
                    remaining_amount = base_val - actual_paid
                    
                    # Find which days still need forecasting (future days only)
                    days_already_paid = set()
                    for actual_date in actual_in_month.index:
                        days_already_paid.add(actual_date.day)
                    
                    # Build list of (day, pct) for days that still need forecasting
                    remaining_days = []
                    for entry in monthly_days:
                        day = entry.get("day")
                        pct = entry.get("percentage", 0.0)

                        # ensure valid day
                        last_day = calendar.monthrange(year, month)[1]
                        day_to_use = min(day, last_day)
                        forecast_date = pd.Timestamp(year, month, day_to_use)
                        if isinstance(forecast_start_date, str):
                            forecast_start_date = pd.to_datetime(forecast_start_date)
                        # Only include if this date is in the future AND hasn't been paid
                        if forecast_date > last_actual and forecast_date > forecast_start_date and day_to_use not in days_already_paid:
                            remaining_days.append((day_to_use, pct, forecast_date))
                    
                    # Calculate total percentage for remaining days
                    total_remaining_pct = sum(pct for _, pct, _ in remaining_days)
                    
                    # Distribute remaining amount proportionally
                    for day_to_use, pct, forecast_date in remaining_days:
                        if total_remaining_pct > 0:
                            allocated_val = remaining_amount * (pct / total_remaining_pct)
                        else:
                            allocated_val = 0.0
                        
                        if forecast_date <= forecast_end_date:
                            new_dates.append(forecast_date)
                            new_vals.append(allocated_val)

                # override future_dates and future_vals
                future_dates = new_dates
                future_vals = new_vals
    
            # index and attach
            future_index = pd.DatetimeIndex(future_dates)
            future_forecast = pd.Series(future_vals, index=future_index)
            combined_series = pd.concat([champion_series, future_forecast])
            # ---- Apply holiday / weekend adjustment (if requested) ----

            def is_holiday_or_weekend(d):
                return d.weekday() >= 5 or d.date() in holiday_set

            if adjustment in ("nextWorkingDay", "previousWorkingDay", "dropTransaction"):
                new_future_index = []
                new_future_forecast_vals = []

                for d, v in zip(future_index, future_forecast):
                    new_d = d
                    if is_holiday_or_weekend(new_d):
                        if adjustment == "dropTransaction":
                            # skip this entry entirely
                            continue
                        elif adjustment == "nextWorkingDay":
                            while is_holiday_or_weekend(new_d):
                                new_d = new_d + timedelta(days=1)
                        elif adjustment == "previousWorkingDay":
                            while is_holiday_or_weekend(new_d):
                                new_d = new_d - timedelta(days=1)

                    new_future_index.append(new_d)
                    new_future_forecast_vals.append(v)

                # replace the original future values with adjusted ones
                future_index = pd.DatetimeIndex(new_future_index)
                future_forecast = pd.Series(new_future_forecast_vals, index=future_index)

                # also update combined series
                combined_series = pd.concat([champion_series, future_forecast])

        # Prepare return (convert index to all present in combined series)
        combined_index = combined_series.index

        return {
            "status": "success",
            "champion_model": champ,
            "champion_composite_type": "simple_composite",
            "metrics": dfm.loc[champ].to_dict(),
            "actual": ts.reindex(combined_index).tolist(),
            # "forecast": combined_series.tolist(),
            # "index": [d.strftime("%Y-%m-%d") for d in combined_index],
            "index": [d.strftime("%Y-%m-%d") for d in future_index],
            "forecast": future_forecast.tolist() if len(future_forecast) > 0 else []
        }

    # ========================================================
    # BIWEEKLY / FORTNIGHTLY CADENCE PATH (NEW, ISOLATED)
    # ========================================================

    # --- Step 1: detect cadence anchor and noise ---
    last_anchor_date, noise_sum = _extract_biweekly_anchor_and_noise(records)

    # --- Step 2: generate cadence dates ---
    future_dates = _generate_biweekly_cadence(
        last_anchor_date,
        forecast_start_date,
        forecast_end_date,
        weekday
    )

    if not future_dates:
        return {"status": "no_future_dates"}

    # --- Step 3: compute base biweekly amount (unchanged SMA/WMA idea) ---
    abs_df = df.copy()
    median_amt = abs_df["amount"].abs().mean()
    abs_df = abs_df[abs_df["amount"].abs() >= 0.15 * median_amt]
    n = min(n_candidates)
    base_amount = abs_df["amount"].iloc[-n:].mean()
    
    # --- Step 4: offset noise into first forecast ---
    future_vals = []
    for i in range(len(future_dates)):
        if i == 0:
            future_vals.append(base_amount + noise_sum)
        else:
            future_vals.append(base_amount)

    return {
        "status": "success",
        "cadence_anchor": last_anchor_date.strftime("%Y-%m-%d"),
        "noise_offset_applied": noise_sum,
        "index": [d.strftime("%Y-%m-%d") for d in future_dates],
        "forecast": future_vals
    }



    