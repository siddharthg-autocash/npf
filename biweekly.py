import pandas as pd
import numpy as np
from datetime import timedelta

WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

def _extract_biweekly_anchor_and_noise(df, pct_threshold=0.15):
    """Identify the last true biweekly anchor and noise amount."""
    # Ensure df is sorted
    df = df.sort_values("date")
    abs_amounts = df["amount"].abs()
    median_amt = abs_amounts.median()
    
    if median_amt == 0 or np.isnan(median_amt):
        return df["date"].iloc[-1], 0.0

    threshold = pct_threshold * median_amt
    df["is_anchor"] = abs_amounts >= threshold
    anchor_rows = df[df["is_anchor"]]

    if anchor_rows.empty:
        return df["date"].iloc[-1], 0.0
    
    last_anchor = anchor_rows["date"].iloc[-1]
    # Noise: Sum of amounts occurring strictly after the last anchor
    noise_sum = df[df["date"] > last_anchor]["amount"].sum()
    return last_anchor, float(noise_sum)

def _generate_cadence(last_anchor, start, end, weekday):
    dates = []
    if isinstance(weekday, str):
        weekday_idx = WEEKDAY_MAP.get(weekday.lower(), 0)
    else:
        weekday_idx = 0 # Default if unknown

    # Step 1: anchor + 14 days
    current = last_anchor + timedelta(days=14)

    # Step 2: align to required weekday
    delta = (weekday_idx - current.weekday()) % 7
    if delta == 0: 
        delta = 7 # ensure forward movement
    current += timedelta(days=delta)

    # Step 3: move into forecast window
    while current < start:
        current += timedelta(days=14)
    
    while current <= end:
        dates.append(current)
        current += timedelta(days=14)
    return dates

def run_biweekly_forecast(df, model_params, forecast_start, forecast_end):
    # 1. Detect Anchor & Noise
    last_anchor, noise = _extract_biweekly_anchor_and_noise(df)
    
    # 2. Generate Dates
    dates = _generate_cadence(
        last_anchor, 
        forecast_start, 
        forecast_end, 
        model_params.get("weekday")
    )
    if not dates:
        return {"status": "no_future_dates"}

    # 3. Compute Base Amount 
    # Logic: Filter absolute amounts > 15% median, then take SMA of last N
    n_candidates = model_params["n_candidates"]
    n = min(n_candidates) if isinstance(n_candidates, list) else 3
    
    abs_df = df.copy()
    median_amt = abs_df["amount"].abs().mean() # Note: Original code used .mean() here on abs series
    # Filter for significant amounts
    sig_df = abs_df[abs_df["amount"].abs() >= 0.15 * median_amt]
    
    if sig_df.empty:
        # Fallback if everything is noise
        base_amount = df["amount"].iloc[-n:].mean()
    else:
        base_amount = sig_df["amount"].iloc[-n:].mean()

    # 4. Forecast with Noise Offset (Only added to first period)
    vals = []
    for i in range(len(dates)):
        if i == 0:
            vals.append(base_amount + noise)
        else:
            vals.append(base_amount)

    return {
        "status": "success",
        "cadence_anchor": last_anchor.strftime("%Y-%m-%d"),
        "noise_offset_applied": noise,
        "index": [d.strftime("%Y-%m-%d") for d in dates],
        "forecast": vals
    }