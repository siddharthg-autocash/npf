import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import last_n_period_model as backend

st.set_page_config(page_title="Forecasting Project", layout="wide")

# Configuration constants
HOLIDAYS = [
    "2025-01-01", "2025-01-20", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-11-11", "2025-11-27", 
    "2025-12-25", "2026-01-01", "2026-01-19"
]

def parse_input_data(text):
    rows = []
    for line in text.split('\n'):
        parts = line.strip().split()
        if len(parts) >= 2:
            rows.append([parts[0], parts[-1]])
    
    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows, columns=["date", "amount"])
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
    return df.dropna().sort_values("date")

# UI Layout
st.title("Time Series Forecasting Demo")

col1, col2 = st.columns([1, 2])

with col1:
    raw_text = st.text_area("Paste Historical Data", height=300)

if raw_text:
    df = parse_input_data(raw_text)
    
    if not df.empty:
        with col2:
            st.subheader("Model Configuration")
            
            freq_option = st.selectbox("Frequency", ["Biweekly", "Weekly", "Monthly", "Quarterly"])
            
            params = {
                "freq": freq_option.lower(),
                "n_candidates": [3, 4, 6],
                "adjustmentOption": "nextWorkingDay"
            }
            
            if freq_option == "Monthly":
                days_input = st.text_input("Target Days (comma separated)", "1")
                if days_input:
                    days_list = [int(x) for x in days_input.split(",") if x.strip().isdigit()]
                    if days_list:
                        pct = 100 / len(days_list)
                        params["monthlyDays"] = [{"day": d, "percentage": pct} for d in days_list]

        if st.button("Run Forecast"):
            # Determine forecast window
            start_date = df["date"].max() + timedelta(days=1)
            end_date = start_date + timedelta(days=180)
            
            try:
                # Call backend logic
                result = backend.run_last_n_forecast(
                    records=df.to_dict(orient="records"),
                    model_params=params,
                    forecast_end_date=end_date,
                    holiday_dates=HOLIDAYS,
                    forecast_start_date=start_date
                )
                
                if result.get("status") == "success":
                    forecast_df = pd.DataFrame({
                        "date": pd.to_datetime(result["index"]),
                        "amount": result["forecast"]
                    })
                    
                    # --- ADD DAY NAMES ---
                    df["day_name"] = df["date"].dt.day_name()
                    forecast_df["day_name"] = forecast_df["date"].dt.day_name()

                    # Visualization
                    fig = go.Figure()
                    
                    # Historical Bar
                    fig.add_trace(go.Bar(
                        x=df["date"], 
                        y=df["amount"], 
                        name="History", 
                        marker_color='orange',
                        customdata=df["day_name"],
                        hovertemplate='%{x|%Y-%m-%d} (%{customdata})<br>Amount: %{y}<extra></extra>'
                    ))
                    
                    # Forecast Bar
                    fig.add_trace(go.Bar(
                        x=forecast_df["date"], 
                        y=forecast_df["amount"], 
                        name="Forecast", 
                        marker_color='red',
                        customdata=forecast_df["day_name"],
                        hovertemplate='%{x|%Y-%m-%d} (%{customdata})<br>Amount: %{y}<extra></extra>'
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Debug Output"):
                        st.json(result)
                else:
                    st.error(f"Model failed: {result.get('status')}")
                    st.write(result)
                    
            except Exception as e:
                st.error(f"Runtime error: {str(e)}")