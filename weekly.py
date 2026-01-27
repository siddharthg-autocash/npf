import pandas as pd
from datetime import timedelta
from core import calculate_champion, sma_flat_forecast_from_actuals, apply_adjustment

def run_weekly_forecast(df, model_params, forecast_start, forecast_end, holiday_set):
    # 1. Aggregate to W-MON
    ts = df.set_index("date")["amount"].resample("W-MON").sum().interpolate(limit_direction="both")
    
    # 2. Model Selection
    n_candidates = model_params["n_candidates"]
    if not isinstance(n_candidates, list): n_candidates = [n_candidates]
    
    champ_n, metrics, _ = calculate_champion(ts, n_candidates)
    if not champ_n:
        return {"status": "no_models", "reason": "no valid SMA models"}
    
    # 3. Generate Future Dates (Weekday Alignment Logic)
    future_dates = []
    weekday_target = model_params.get("weekday") # e.g., "Friday"
    
    # Start looking from last actual date
    current = ts.index[-1]
    
    # Logic from original script:
    # If first step and weekday is set, find next matching weekday strictly after last actual
    first_step = True
    
    while True:
        if weekday_target:
            if first_step:
                current = current + timedelta(days=1)
                while current.strftime("%A") != weekday_target:
                    current += timedelta(days=1)
            else:
                current = current + timedelta(days=7)
                while current.strftime("%A") != weekday_target:
                    current += timedelta(days=1)
        else:
            # No specific weekday, just add 7 days (standard weekly)
            current = current + timedelta(days=7)
            
        if current > forecast_end:
            break
            
        if current >= forecast_start:
            future_dates.append(current)
            
        first_step = False
        
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