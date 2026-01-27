import pandas as pd
import biweekly
import monthly
import weekly
import quarterly

def run_last_n_forecast(
    records,
    model_params,
    forecast_end_date,
    holiday_dates,
    forecast_start_date
):
    if not records:
        return {"status": "no_data"}

    # --- Preprocessing (Common) ---
    forecast_end_date = pd.to_datetime(forecast_end_date)
    forecast_start_date = pd.to_datetime(forecast_start_date)
    holiday_set = {pd.to_datetime(h).date() for h in holiday_dates or []}

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    
    # Group by date to handle multiple transactions same day
    df = df.groupby("date", as_index=False)["amount"].sum()

    freq = model_params.get("freq", "").lower()

    # --- Dispatcher ---
    if freq in ("biweekly", "fortnightly"):
        return biweekly.run_biweekly_forecast(
            df, model_params, forecast_start_date, forecast_end_date
        )
    
    elif freq == "monthly":
        return monthly.run_monthly_forecast(
            df, model_params, forecast_start_date, forecast_end_date, holiday_set
        )
        
    elif freq == "weekly":
        return weekly.run_weekly_forecast(
            df, model_params, forecast_start_date, forecast_end_date, holiday_set
        )
        
    elif freq == "quarterly":
        return quarterly.run_quarterly_forecast(
            df, model_params, forecast_start_date, forecast_end_date, holiday_set
        )
    
    else:
        # Fallback to weekly if unknown
        return weekly.run_weekly_forecast(
            df, model_params, forecast_start_date, forecast_end_date, holiday_set
        )