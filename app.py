import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
# IMPORT YOUR BACKEND
import last_n_period_model as backend

st.set_page_config(page_title="forecats", layout="wide")

# Standard Holidays
HOLIDAY_DATES = [
    "2025-01-01", "2025-01-20", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-11-11", "2025-11-27", 
    "2025-12-25", "2026-01-01", "2026-01-19"
]

def parse_input(raw_text):
    data = []
    for line in raw_text.split('\n'):
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0][0].isdigit():
            data.append([parts[0], parts[-1]])
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data, columns=["date", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"])
    return df.sort_values("date")

st.title("💸 Forecats")
st.markdown("This UI has **no logic**. It sends data to `last_n_period_model.py`.")

default_input = """
2025-11-26T00:00:00		-6395029.03
2025-11-28T00:00:00		   24.24
2025-12-01T00:00:00		  -37242.68
2025-12-12T00:00:00		-6467417.19
2025-12-30T00:00:00		-6651171.38
2026-01-05T00:00:00		   -1655.79
2026-01-06T00:00:00		  -68105.95
2026-01-07T00:00:00		-839934.30
"""

col1, col2 = st.columns([1, 2])
with col1:
    raw_text = st.text_area("Input Data", value=default_input, height=300)

if raw_text:
    df = parse_input(raw_text)
    if not df.empty:
        with col2:
            st.subheader("Settings")
            # MANUAL FREQUENCY SELECTOR
            freq = st.selectbox("Frequency", ["Biweekly", "Weekly", "Monthly", "Quarterly"])
            
            # PARAMS
            model_params = {
                "freq": freq.lower(), 
                "n_candidates": [3, 4, 6], 
                "adjustmentOption": "nextWorkingDay"
            }
            
            # Monthly specific inputs
            if freq == "Monthly":
                days = st.text_input("Monthly Days (e.g. 1, 10, 20)", "")
                if days:
                    d_list = [int(x.strip()) for x in days.split(",")]
                    model_params["monthlyDays"] = [{"day": d, "percentage": 100/len(d_list)} for d in d_list]

        if st.button("Run Backend Logic"):
            # PREPARE PAYLOAD
            records = df.to_dict(orient="records")
            start = df["date"].max() + timedelta(days=1)
            end = start + timedelta(days=180)

            try:
                # --- CALL BACKEND ---
                result = backend.run_last_n_forecast(
                    records=records,
                    model_params=model_params,
                    forecast_end_date=end,
                    holiday_dates=HOLIDAY_DATES,
                    forecast_start_date=start
                )
                
                # DISPLAY RESULTS
                if result.get("status") == "success":
                    # Plot
                    fcst_df = pd.DataFrame({
                        "date": pd.to_datetime(result["index"]),
                        "amount": result["forecast"]
                    })
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df["date"], y=df["amount"], name="Actual", marker_color='#b71c1c'))
                    fig.add_trace(go.Bar(x=fcst_df["date"], y=fcst_df["amount"], name="Forecast", marker_color='#ef5350'))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"Backend Success! Mode: {freq}")
                    with st.expander("View JSON Response"):
                        st.json(result)
                else:
                    st.error(f"Backend Failed: {result}")
            except Exception as e:
                st.error(f"Python Error: {e}")