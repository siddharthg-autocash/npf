# forecastFiles/quarterly.py
import pandas as pd
from forecastFiles.core import calculate_champion, sma_flat_forecast_from_actuals, apply_adjustment

def run_quarterly_forecast(df, model_params, forecast_start, forecast_end, holiday_set):
    # 1. Aggregate to Quarter end labels (Q)
    ts = df.set_index("date")["amount"].resample("Q").sum().interpolate(limit_direction="both")

    # 2. Model Selection
    n_candidates = model_params.get("n_candidates", [3])
    if not isinstance(n_candidates, list):
        n_candidates = [n_candidates]

    champ_n, metrics, _ = calculate_champion(ts, n_candidates)
    if not champ_n:
        return {"status": "no_models", "reason": "no valid SMA models"}

    # 3. Generate Future Dates
    future_dates = []
    # Anchor from last actual transaction
    last_actual_date = df["date"].max()
    # Start from next quarter end after last ts label
    curr = ts.index[-1] + pd.offsets.QuarterEnd(1)

    while curr <= forecast_end:
        if curr >= forecast_start:
            future_dates.append(curr)
        curr += pd.offsets.QuarterEnd(1)

    # 4. Forecast Values
    # Exclude partial last quarter if last_actual_date is before quarter end
    ts_for_forecast = ts
    try:
        last_q_label = ts.index[-1]
        # quarter end is last_q_label (resample with Q yields quarter-end labels)
        if last_actual_date < last_q_label and len(ts) >= 2:
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
        "forecast": future_vals
    }
