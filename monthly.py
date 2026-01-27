import pandas as pd
import numpy as np
import calendar
from core import calculate_champion, sma_flat_forecast_from_actuals, apply_adjustment

def run_monthly_forecast(df, model_params, forecast_start, forecast_end, holiday_set):
    # 1. Resample to Monthly Start (MS) for Modeling
    ts = df.set_index("date")["amount"].resample("MS").sum().interpolate(limit_direction="both")
    
    # 2. Model Selection (SMA Only)
    n_candidates = model_params["n_candidates"]
    if not isinstance(n_candidates, list): n_candidates = [n_candidates]
    
    champ_n, metrics, _ = calculate_champion(ts, n_candidates)
    if not champ_n:
        return {"status": "no_models", "reason": "no valid SMA models"}

    # 3. Prepare for Forecasting
    monthly_days = model_params.get("monthlyDays") # List of {day: 1, percentage: 50}
    day_of_month = model_params.get("day_of_month")
    
    last_actual_ts = ts.index[-1] # The MS timestamp
    # Original logic: last_actual is the index (timestamp)
    
    # Determine the "Base" value for the month
    # FIX from original: If monthly multi-day, exclude current partial month from average
    if monthly_days:
        # last_actual_ts is likely the first of the current month.
        # We want strict past data for the average.
        ts_for_avg = ts[ts.index < last_actual_ts]
    else:
        ts_for_avg = ts
        
    # Compute the flat monthly total
    if ts_for_avg.empty:
         base_monthly_val = 0.0
    else:
         base_monthly_val = sma_flat_forecast_from_actuals(ts_for_avg, champ_n, steps=1)[0]
    
    future_dates = []
    future_vals = []
    
    # 4. Generate Dates & Values Loop
    # We iterate month by month starting from the month of the last actual
    curr_month_start = last_actual_ts 
    
    while True:
        year, month = curr_month_start.year, curr_month_start.month
        
        # --- PATH A: Multi-Day Split (Complex Logic) ---
        if monthly_days:
            month_start = pd.Timestamp(year, month, 1)
            last_day_num = calendar.monthrange(year, month)[1]
            month_end = pd.Timestamp(year, month, last_day_num)
            
            # Check actuals paid in this specific month
            actual_in_month = df[(df["date"] >= month_start) & (df["date"] <= month_end)]["amount"].sum()
            remaining_amount = base_monthly_val - actual_in_month
            
            # Which days are already paid?
            days_already_paid = set(df[(df["date"] >= month_start) & (df["date"] <= month_end)]["date"].dt.day)
            
            # Build candidates
            candidates = []
            for entry in monthly_days:
                d = entry.get("day")
                pct = float(entry.get("percentage", 0.0))
                if d is None: continue
                
                day_to_use = min(d, last_day_num)
                dt = pd.Timestamp(year, month, day_to_use)
                
                # Logic: Must be in future window AND not already paid
                if dt > df["date"].max() and dt > forecast_start and day_to_use not in days_already_paid:
                    candidates.append((dt, pct))
            
            # Distribute remaining amount
            total_remaining_pct = sum(c[1] for c in candidates)
            for dt, pct in candidates:
                if dt <= forecast_end:
                    val = remaining_amount * (pct / total_remaining_pct) if total_remaining_pct > 0 else 0
                    future_dates.append(dt)
                    future_vals.append(val)

        # --- PATH B: Standard Monthly (Single Date) ---
        else:
            # We look at the "Next Month" usually, unless we are filling the current month
            # The original logic sets 'next_month = current + MonthBegin(1)'
            # But let's stick to the simpler stepping logic provided in the original else block
            
            # Calculate candidate date for this month
            last_day_num = calendar.monthrange(year, month)[1]
            if day_of_month:
                day_to_use = min(day_of_month, last_day_num)
            else:
                day_to_use = 1 # Default to MS
            
            dt = pd.Timestamp(year, month, day_to_use)
            
            # Only add if it's strictly in the future window
            if dt > df["date"].max() and dt > forecast_start and dt <= forecast_end:
                future_dates.append(dt)
                future_vals.append(base_monthly_val)

        # Move to next month
        curr_month_start = curr_month_start + pd.offsets.MonthBegin(1)
        if curr_month_start > forecast_end:
            break
            
    # Sort results
    if future_dates:
        # Zip, sort by date, unzip
        zipped = sorted(zip(future_dates, future_vals), key=lambda x: x[0])
        future_dates, future_vals = zip(*zipped)

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
        "forecast": list(final_vals)
    }