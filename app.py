import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import calendar
import re
import numpy as np

# ==========================================
# 1. CONFIGURATION & HOLIDAYS
# ==========================================
st.set_page_config(page_title="Auto Forecast", layout="wide")

# Standard US Holidays (Extended for 2025-2026)
# Used to shift forecast dates if they land on these days
HOLIDAYS = {
    pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-01-20").date(),
    pd.Timestamp("2025-05-26").date(), pd.Timestamp("2025-06-19").date(),
    pd.Timestamp("2025-07-04").date(), pd.Timestamp("2025-09-01").date(),
    pd.Timestamp("2025-11-11").date(), pd.Timestamp("2025-11-27").date(),
    pd.Timestamp("2025-12-25").date(),
    pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-19").date(),
}

def adjust_to_working_day(dt):
    """
    Moves a date forward to the next working day 
    if it falls on a Saturday (5), Sunday (6), or Holiday.
    """
    while dt.weekday() >= 5 or dt.date() in HOLIDAYS:
        dt += timedelta(days=1)
    return dt

# ==========================================
# 2. DATA PARSING
# ==========================================
def parse_markdown_data(raw_text):
    """
    Parses specific '## Header' format from test_data.md
    """
    sections = {}
    current_section = None
    current_data = []
    
    lines = raw_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # New Section Header
        if line.startswith("##"):
            # Save previous section
            if current_section and current_data:
                df = pd.DataFrame(current_data, columns=["CF Date", "Actuals"])
                df["CF Date"] = pd.to_datetime(df["CF Date"])
                df["Actuals"] = pd.to_numeric(df["Actuals"])
                sections[current_section] = df.sort_values("CF Date")
            
            current_section = line.replace("##", "").strip()
            current_data = []
            
        # Data Row (YYYY-MM-DD ...)
        elif re.match(r'^\d{4}-\d{2}-\d{2}', line):
            parts = line.split()
            # Assuming Date is first, Amount is last
            if len(parts) >= 2:
                current_data.append([parts[0], parts[-1]])

    # Save last section
    if current_section and current_data:
        df = pd.DataFrame(current_data, columns=["CF Date", "Actuals"])
        df["CF Date"] = pd.to_datetime(df["CF Date"])
        df["Actuals"] = pd.to_numeric(df["Actuals"])
        sections[current_section] = df.sort_values("CF Date")
        
    return sections

# ==========================================
# 3. AUTO-DETECTION LOGIC
# ==========================================
def detect_pattern(df):
    """
    Analyzes date gaps to determine the best forecast strategy.
    Returns: (Mode String, Explanation String, Extra Params)
    """
    df = df.sort_values("CF Date")
    dates = df["CF Date"]
    
    if len(dates) < 3:
        return "Biweekly", "Not enough data (defaulting)", {}

    # Calculate gap in days between transactions
    gaps = dates.diff().dt.days.dropna()
    median_gap = gaps.median()
    
    # 1. Check for Monthly Multi-Day Pattern (e.g. 1st, 10th, 20th)
    # If the median gap is small (e.g., 7-12 days) but it's NOT a consistent weekly (7) or biweekly (14)
    # We check if specific days of the month repeat.
    
    days_of_month = dates.dt.day
    # Count how often days appear. 
    # We group nearby days (e.g., 1, 2, 3 -> Group 1) to handle weekends.
    day_counts = {}
    for d in days_of_month:
        found_group = False
        for k in day_counts.keys():
            if abs(k - d) <= 2: # +/- 2 days tolerance
                day_counts[k].append(d)
                found_group = True
                break
        if not found_group:
            day_counts[d] = [d]
            
    # If we have distinct clusters that repeat
    significant_clusters = [int(np.min(v)) for k, v in day_counts.items() if len(v) >= 2]
    significant_clusters.sort()
    
    # Heuristic: If median gap is irregular (8-12 days) and we found monthly clusters
    if 7 < median_gap < 13 and len(significant_clusters) >= 2:
        return "Monthly Pattern", f"Detected Monthly Multi-Day Pattern (Days: {significant_clusters})", {"days": significant_clusters}

    # 2. Check Standard Cadences
    if 6 <= median_gap <= 8:
        return "Weekly", f"Detected Weekly Pattern (Every ~{int(median_gap)} days)", {}
    
    if 13 <= median_gap <= 15:
        return "Biweekly", f"Detected Biweekly Pattern (Every ~{int(median_gap)} days)", {}
        
    # Default Fallback
    return "Biweekly", "Pattern Unclear (Defaulting to Biweekly)", {}

# ==========================================
# 4. FORECAST ENGINE
# ==========================================
def generate_forecast(df, mode, params, months_to_project=6):
    last_actual = df["CF Date"].max()
    
    # Calculate Forecast Amount (Avg of last 6)
    base_amount = df["Actuals"].tail(6).mean()
    
    forecast_rows = []
    
    # --- STRATEGY: MONTHLY MULTI-DAY ---
    if mode == "Monthly Pattern":
        target_days = params.get("days", [1]) # Default to 1st if missing
        
        # Start from current month
        curr = last_actual.replace(day=1)
        
        for _ in range(months_to_project):
            year, month = curr.year, curr.month
            last_day = calendar.monthrange(year, month)[1]
            
            for d in target_days:
                # Handle end of month
                day_to_use = min(d, last_day)
                
                # Create Base Date
                dt = pd.Timestamp(year, month, day_to_use)
                
                # Apply Holiday Adjustment (Next Working Day)
                adj_dt = adjust_to_working_day(dt)
                
                # Only add if it's strictly in the future
                if adj_dt > last_actual:
                    forecast_rows.append({"CF Date": adj_dt, "Forecast": base_amount})
            
            # Move to next month
            curr = curr + pd.offsets.MonthBegin(1)

    # --- STRATEGY: WEEKLY / BIWEEKLY ---
    else:
        step_days = 7 if mode == "Weekly" else 14
        curr = last_actual + timedelta(days=step_days)
        
        # Generate roughly same count (6 months * 2 or 4)
        count = months_to_project * (4 if step_days == 7 else 2)
        
        for _ in range(count):
            # Apply Holiday Adjustment even for simple cadences
            adj_dt = adjust_to_working_day(curr)
            
            forecast_rows.append({"CF Date": adj_dt, "Forecast": base_amount})
            curr += timedelta(days=step_days)

    return pd.DataFrame(forecast_rows)

# ==========================================
# 5. VISUALIZATION
# ==========================================
def plot_forecast(df_actual, df_forecast, title):
    fig = go.Figure()
    
    # Actuals (Dark Red)
    fig.add_trace(go.Bar(
        x=df_actual["CF Date"], y=df_actual["Actuals"],
        name="Actual Amt(-)", marker_color='#b71c1c'
    ))
    
    # Forecast (Light Red)
    fig.add_trace(go.Bar(
        x=df_forecast["CF Date"], y=df_forecast["Forecast"],
        name="Fcst Amt(-)", marker_color='#ef5350'
    ))
    
    fig.update_layout(
        title=f"Forecast: {title}",
        xaxis_title="Date", 
        yaxis_title="Amount",
        template="plotly_white",
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right")
    )
    return fig

# ==========================================
# 6. MAIN APP LAYOUT
# ==========================================
st.title("💸 Auto-Forecast Engine")
st.markdown("Paste your data below. The system will **automatically detect** if it's Weekly, Biweekly, or a specific Monthly pattern (e.g. 1st, 10th, 20th).")

# Default Data (PSSI Example)
default_input = """
## PSSI Write Express
CF Date				Actuals
2025-10-01T00:00:00		-33845.21
2025-10-10T00:00:00		-32312.67
2025-10-20T00:00:00		-27618.60
2025-11-03T00:00:00		-46004.29
2025-11-10T00:00:00		-24145.73
2025-11-20T00:00:00		-40496.01
2025-12-01T00:00:00		-25659.14
2025-12-10T00:00:00		-27820.15
2025-12-22T00:00:00		-31324.00
2026-01-02T00:00:00		-19666.58
"""

col1, col2 = st.columns([1, 2])

with col1:
    raw_text = st.text_area("Input Data", value=default_input, height=300)

if raw_text:
    # 1. Parse
    datasets = parse_markdown_data(raw_text)
    
    if datasets:
        # 2. Select Dataset
        selected_name = st.selectbox("Select Account", list(datasets.keys()))
        df_selected = datasets[selected_name]
        
        # 3. Detect Pattern
        auto_mode, auto_explanation, auto_params = detect_pattern(df_selected)
        
        with col2:
            st.info(f"🔍 **Auto-Detection Analysis:**\n{auto_explanation}")
        
        # 4. User Override Option
        st.subheader("Configuration")
        use_manual = st.checkbox("Override Auto-Detection?", value=False)
        
        if use_manual:
            final_mode = st.selectbox("Manual Frequency", ["Biweekly", "Weekly", "Monthly Pattern"])
            if final_mode == "Monthly Pattern":
                # Let user define days if manual
                days_input = st.text_input("Enter days (comma separated)", "1, 10, 20")
                final_params = {"days": [int(x.strip()) for x in days_input.split(",")]}
            else:
                final_params = {}
        else:
            final_mode = auto_mode
            final_params = auto_params

        # 5. Run Forecast
        df_fcst = generate_forecast(df_selected, final_mode, final_params)
        
        # 6. Visualize
        st.divider()
        st.plotly_chart(plot_forecast(df_selected, df_fcst, selected_name), use_container_width=True)
        
        # 7. Show Data Table (to verify Jan 12th logic)
        with st.expander("View Forecast Data Details"):
            st.dataframe(df_fcst)
            
    else:
        st.error("No valid data found. Check format.")