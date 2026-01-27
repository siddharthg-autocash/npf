import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta
import re

# ==========================================
# 1. PARSING LOGIC (Handles your specific format)
# ==========================================
def parse_markdown_data(raw_text):
    """
    Parses the copy-pasted text from test_data.md into a dictionary of DataFrames.
    Splits by '## Header' sections.
    """
    sections = {}
    current_section = None
    current_data = []

    lines = raw_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect Header (e.g., ## RM 401k)
        if line.startswith("##"):
            # Save previous section if exists
            if current_section and current_data:
                df = pd.DataFrame(current_data, columns=["CF Date", "Actuals"])
                df["CF Date"] = pd.to_datetime(df["CF Date"])
                df["Actuals"] = pd.to_numeric(df["Actuals"])
                sections[current_section] = df
            
            # Start new section
            current_section = line.replace("##", "").strip()
            current_data = []
        
        # Detect Data Rows (Date followed by Number)
        # Regex looks for: YYYY-MM-DD... whitespace ... Number
        elif re.match(r'^\d{4}-\d{2}-\d{2}', line):
            parts = line.split()
            # Assuming first part is date, last part is amount
            if len(parts) >= 2:
                date_str = parts[0]
                amount_str = parts[-1]
                current_data.append([date_str, amount_str])

    # Save the last section
    if current_section and current_data:
        df = pd.DataFrame(current_data, columns=["CF Date", "Actuals"])
        df["CF Date"] = pd.to_datetime(df["CF Date"])
        df["Actuals"] = pd.to_numeric(df["Actuals"])
        sections[current_section] = df
        
    return sections

# ==========================================
# 2. FORECAST LOGIC (Biweekly Anchor)
# ==========================================
def run_simple_biweekly_forecast(df, periods=6):
    """
    Simplified version of the logic we discussed:
    1. Find Anchor (Heartbeat).
    2. Project forward 14 days at a time.
    """
    df = df.sort_values("CF Date")
    
    # 1. Identify Anchors (Median Threshold Logic)
    abs_amounts = df["Actuals"].abs()
    median_amt = abs_amounts[abs_amounts > 0].median()
    threshold = 0.15 * median_amt
    
    # Filter to get "Anchor" rows
    df["is_anchor"] = abs_amounts >= threshold
    anchor_rows = df[df["is_anchor"]]
    
    if anchor_rows.empty:
        last_anchor_date = df["CF Date"].max()
        base_amount = df["Actuals"].mean()
    else:
        last_anchor_date = anchor_rows["CF Date"].max()
        # Average of last 3 anchors for the forecast amount
        base_amount = anchor_rows["Actuals"].tail(3).mean()

    # 2. Generate Forecast Dates & Values
    forecast_data = []
    current_date = last_anchor_date
    
    # Skip forward until we are past the last actual date
    last_actual_date = df["CF Date"].max()
    while current_date <= last_actual_date:
        current_date += timedelta(days=14)
        
    # Generate 'periods' number of future payments
    for i in range(periods):
        forecast_data.append({
            "CF Date": current_date,
            "Forecast": base_amount
        })
        current_date += timedelta(days=14)
        
    return pd.DataFrame(forecast_data)

# ==========================================
# 3. PLOTTING (Matches Screenshot Style)
# ==========================================
def plot_chart(df_actual, df_forecast, title):
    fig = go.Figure()

    # ACTUALS (Dark Red)
    fig.add_trace(go.Bar(
        x=df_actual["CF Date"],
        y=df_actual["Actuals"],
        name="Actual Amt(-)",
        marker_color='#b71c1c', # Deep Red
        width=86400000 * 0.5 # Fix bar width slightly (milliseconds)
    ))

    # FORECAST (Light Red/Pinkish)
    fig.add_trace(go.Bar(
        x=df_forecast["CF Date"],
        y=df_forecast["Forecast"],
        name="Fcst Amt(-)",
        marker_color='#e57373', # Light Red
        width=86400000 * 0.5 
    ))

    # Vertical Line separator
    last_actual = df_actual["CF Date"].max()
    fig.add_vline(x=last_actual.timestamp() * 1000, line_width=1, line_dash="dash", line_color="grey")

    fig.update_layout(
        title=f"Forecast Chart: {title}",
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True, 
            gridcolor='#f0f0f0',
            tickformat="%d %b\n%Y" # Format like: 01 Oct 2025
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinecolor='black'
        ),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# ==========================================
# 4. MAIN APP INTERFACE
# ==========================================
st.set_page_config(layout="wide")

st.title("Auto Forecast Visualizer")
st.markdown("Paste your `test_data.md` content below to run the simulation.")

# DEFAULT DATA (RM 401k from your file)
default_text = """
## RM 401k
CF Date				Actuals
2025-05-05T00:00:00		-322554.66
2025-05-12T00:00:00		-1158.2
2025-05-16T00:00:00		-296310.77
2025-05-30T00:00:00		-309250.78
2025-06-16T00:00:00		-290403.16
2025-06-30T00:00:00		-288412.53
2025-07-15T00:00:00		-289742.78
2025-07-31T00:00:00		-317437.32
2025-08-19T00:00:00		-296951.91
2025-09-02T00:00:00		-298858.75
2025-09-15T00:00:00		-291171.47
2025-10-01T00:00:00		-289801
2025-10-16T00:00:00		-289045.37
2025-10-31T00:00:00		-310369.53
2025-11-17T00:00:00		-290027.61
2025-12-01T00:00:00		-297899.41
2025-12-15T00:00:00		-286806.33
2025-12-30T00:00:00		-303999.3
2026-01-02T00:00:00		-7379.49
"""

# Input Area
raw_input = st.text_area("Input Data", value=default_text, height=200)

if raw_input:
    # 1. Parse
    datasets = parse_markdown_data(raw_input)
    
    if not datasets:
        st.error("No valid data found. Ensure format matches '## Name' followed by 'Date Value'.")
    else:
        # 2. Select Dataset
        option = st.selectbox("Select Dataset to Forecast", list(datasets.keys()))
        df_selected = datasets[option]
        
        # 3. Forecast
        df_fcst = run_simple_biweekly_forecast(df_selected)
        
        # 4. Visualize
        st.subheader(f"Analysis: {option}")
        
        # Metrics Row
        col1, col2, col3 = st.columns(3)
        avg_amt = df_fcst["Forecast"].iloc[0] if not df_fcst.empty else 0
        col1.metric("Latest Actual", df_selected["CF Date"].max().strftime("%Y-%m-%d"))
        col2.metric("Next Forecast Date", df_fcst["CF Date"].min().strftime("%Y-%m-%d"))
        col3.metric("Projected Amount", f"${avg_amt:,.2f}")

        # Plot
        fig = plot_chart(df_selected, df_fcst, option)
        st.plotly_chart(fig, use_container_width=True)

        # Show Data
        with st.expander("See Calculation Data"):
            c1, c2 = st.columns(2)
            c1.dataframe(df_selected)
            c2.dataframe(df_fcst)