# forecastFiles/weekly.py
import pandas as pd
from datetime import timedelta
from forecastFiles.core import calculate_champion, sma_flat_forecast_from_actuals, apply_adjustment, WEEKDAY_MAP

def run_weekly_forecast(df, model_params, forecast_start, forecast_end, holiday_set):
    # 1. Aggregate to W-MON (weeks labeled by Monday)
    ts = df.set_index("date")["amount"].resample("W-MON").sum().interpolate(limit_direction="both")

    # 2. Model Selection
    n_candidates = model_params.get("n_candidates", [3])
    if not isinstance(n_candidates, list):
        n_candidates = [n_candidates]

    champ_n, metrics, _ = calculate_champion(ts, n_candidates)
    if not champ_n:
        return {"status": "no_models", "reason": "no valid SMA models"}

    # 3. Generate Future Dates (anchor from last actual transaction)
    future_dates = []
    weekday_target = model_params.get("weekday")  # e.g., "Wednesday"

    if weekday_target:
        target_idx = WEEKDAY_MAP.get(weekday_target.lower(), 0)
    else:
        target_idx = 0  # default Monday (resample)

    # Anchor: last actual transaction date (not ts.index[-1])
    last_actual_date = df["date"].max()

    # Find strictly NEXT occurrence of target weekday
    delta = (target_idx - last_actual_date.weekday()) % 7
    days_to_add = 7 if delta == 0 else delta
    current = last_actual_date + timedelta(days=days_to_add)

    # Loop forward weekly
    while current <= forecast_end:
        if current >= forecast_start:
            future_dates.append(pd.to_datetime(current))
        current += timedelta(days=7)

    # 4. Forecast Values
    # Exclude partial last week from base if necessary
    ts_for_forecast = ts
    try:
        last_week_label = ts.index[-1]  # Monday label
        # end of that week is label + 6 days
        if last_actual_date < (last_week_label + timedelta(days=6)):
            # partial week -> drop last label if we have at least 2 points
            if len(ts) >= 2:
                ts_for_forecast = ts.iloc[:-1]
    except Exception:
        ts_for_forecast = ts

    future_vals = sma_flat_forecast_from_actuals(ts_for_forecast, champ_n, steps=len(future_dates))

    # 5. Holiday Adjustments
    final_dates, final_vals = apply_adjustment(
        future_dates, future_vals, model_params.get("adjustmentOption"), holiday_set
    )

    return {
        "status": "success",
        "champion_model": f"SMA_{champ_n}",
        "metrics": metrics,
        "actual": ts.tolist(),
        "index": [d.strftime("%Y-%m-%d") for d in final_dates],
        "forecast": final_vals
    }
