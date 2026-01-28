# last_n_period_model.py
import pandas as pd
from datetime import timedelta

import forecastFiles.biweekly as biweekly
import forecastFiles.monthly as monthly
import forecastFiles.weekly as weekly
import forecastFiles.quarterly as quarterly
from forecastFiles.core import apply_two_sigma_filter

def run_last_n_forecast(
    records,
    model_params,
    forecast_end_date,
    holiday_dates,
    forecast_start_date
):
    """
    records: list/dict rows with 'date' and 'amount'
    model_params: dict containing freq, n_candidates, adjustmentOption, weekday, etc.
    forecast_end_date, forecast_start_date: string or datetime-like
    holiday_dates: iterable of date strings
    """
    if not records:
        return {"status": "no_data"}

    # --- Preprocessing (Common) ---
    forecast_end_date = pd.to_datetime(forecast_end_date)
    forecast_start_date = pd.to_datetime(forecast_start_date)
    holiday_set = {pd.to_datetime(h).date() for h in (holiday_dates or [])}

    # Raw dataframe (unfiltered) for anchor detection
    df_raw = pd.DataFrame(records)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw["amount"] = pd.to_numeric(df_raw["amount"], errors="coerce")
    df_raw = df_raw.dropna(subset=["date"]).sort_values("date")

    # ALWAYS apply 2-sigma outlier filter (automatic) — but only for base computations
    sigma = 2.0
    df_filtered = apply_two_sigma_filter(df_raw, sigma)
    # If filter removed everything, fallback to original df_raw to avoid empty streams
    if df_filtered.empty:
        df_filtered = df_raw.copy()

    # Group by date to handle multiple transactions same day
    df_grouped = df_filtered.groupby("date", as_index=False)["amount"].sum()
    # Also keep grouped raw (unfiltered) for anchor detection/noise calculation
    df_raw_grouped = df_raw.groupby("date", as_index=False)["amount"].sum()

    freq = model_params.get("freq", "").lower()

    # --- Dispatcher ---
    if freq in ("biweekly", "fortnightly"):
        # pass raw grouped df too for anchor detection
        return biweekly.run_biweekly_forecast(
            df_grouped, model_params, forecast_start_date, forecast_end_date, holiday_set, raw_df=df_raw_grouped
        )

    elif freq == "monthly":
        return monthly.run_monthly_forecast(
            df_grouped, model_params, forecast_start_date, forecast_end_date, holiday_set
        )

    elif freq == "weekly":
        return weekly.run_weekly_forecast(
            df_grouped, model_params, forecast_start_date, forecast_end_date, holiday_set
        )

    elif freq == "quarterly":
        return quarterly.run_quarterly_forecast(
            df_grouped, model_params, forecast_start_date, forecast_end_date, holiday_set
        )

    else:
        # Fallback to weekly if unknown
        return weekly.run_weekly_forecast(
            df_grouped, model_params, forecast_start_date, forecast_end_date, holiday_set
        )

