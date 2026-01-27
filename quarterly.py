import pandas as pd
from core import calculate_champion, sma_flat_forecast_from_actuals, apply_adjustment

def run_quarterly_forecast(df, model_params, forecast_start, forecast_end, holiday_set):
    # 1. Aggregate
    ts = df.set_index("date")["amount"].resample("Q").sum().interpolate(limit_direction="both")
    
    # 2. Model Selection
    n_candidates = model_params["n_candidates"]
    if not isinstance(n_candidates, list): n_candidates = [n_candidates]
    
    champ_n, metrics, _ = calculate_champion(ts, n_candidates)
    if not champ_n:
        return {"status": "no_models", "reason": "no valid SMA models"}
        
    # 3. Generate Future Dates
    future_dates = []
    curr = ts.index[-1] + pd.offsets.QuarterEnd(1)
    
    while curr <= forecast_end:
        if curr >= forecast_start:
            future_dates.append(curr)
        curr += pd.offsets.QuarterEnd(1)

    # 4. Forecast Values
    future_vals = sma_flat_forecast_from_actuals(ts, champ_n, steps=len(future_dates))
    
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