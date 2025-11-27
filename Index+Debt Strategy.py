import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import date

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Multi-Asset Strategy Backtester", layout="wide")

# --- FILE LOADING LOGIC ---
DEFAULT_PATH = r"D:\MIRA Money\Data I've Analyzed\Individual Sheets\Index Data for Strategy.xlsx"

@st.cache_data
def load_data(uploaded_file=None):
    df = None
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            return None
    elif os.path.exists(DEFAULT_PATH):
        try:
            if DEFAULT_PATH.endswith('.csv'):
                df = pd.read_csv(DEFAULT_PATH)
            else:
                df = pd.read_excel(DEFAULT_PATH)
        except Exception as e:
            st.warning(f"Found default file but could not load: {e}")
            return None
    else:
        return None

    if df is not None:
        # Clean Data
        df.columns = df.columns.str.strip()
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.rename(columns={date_col: 'Date'})
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
        # Forward fill to handle holidays/gaps, drop remaining NaNs
        df = df.ffill().dropna()
        
    return df

# --- BACKTEST ENGINE (MULTI-ASSET) ---
def run_portfolio_backtest(df_slice, weights_dict, rebalance_freq):
    """
    weights_dict: {'Asset_Name': 0.50, 'Asset_Name_2': 0.30, ...} (Sum should be 1.0)
    """
    assets = list(weights_dict.keys())
    
    # Calculate daily returns for selected assets
    returns = df_slice[assets].pct_change().dropna()
    
    if returns.empty:
        return None

    # Align dates
    date_index = returns.index
    n_days = len(date_index)
    
    # Initial setup
    current_value = 100.0
    
    # Current value of each component [Asset1_Val, Asset2_Val...]
    # Initialize based on weights
    component_values = np.array([current_value * weights_dict[a] for a in assets])
    
    # Store history
    history = [current_value]
    dates = [date_index[0]]
    
    # Convert returns to numpy array for speed: Shape (Rows, Cols)
    ret_values = returns.values 
    
    # Map Rebalance Frequency
    is_daily = rebalance_freq == "Daily"
    is_monthly = rebalance_freq == "Monthly"
    is_yearly = rebalance_freq == "Yearly"
    is_never = "Never" in rebalance_freq
    
    # Target weights array
    target_weights = np.array([weights_dict[a] for a in assets])

    # Loop through days (starting from 2nd day relative to returns index)
    for i in range(1, n_days):
        today = date_index[i]
        prev_date = date_index[i-1]
        
        # 1. Apply Returns to each component individually (Drift)
        # component_new = component_old * (1 + daily_return)
        component_values = component_values * (1 + ret_values[i])
        
        total_val = np.sum(component_values)
        
        # 2. Check Rebalance Trigger
        rebalance = False
        if is_daily:
            rebalance = True
        elif is_monthly and today.month != prev_date.month:
            rebalance = True
        elif is_yearly and today.year != prev_date.year:
            rebalance = True
            
        # 3. Rebalance if needed
        if rebalance and not is_never:
            # Reset components to Total * Target_Weight
            component_values = total_val * target_weights
            
        history.append(total_val)
        dates.append(today)
        
    return pd.Series(history, index=dates, name="Strategy")

# --- SIDEBAR & SETUP ---
st.sidebar.header("Configuration")

# 1. Load Data
uploaded_file = st.sidebar.file_uploader("Upload Data", type=['xlsx', 'csv'])
df_raw = load_data(uploaded_file)

if df_raw is None:
    st.info("Please upload data or check the default path.")
    st.stop()

all_cols = df_raw.columns.tolist()

# 2. Portfolio Construction
st.sidebar.subheader("Portfolio Composition")

# Multi-select for assets
selected_assets = st.sidebar.multiselect("Select All Assets (Equity & Debt)", all_cols, default=all_cols[:2] if len(all_cols)>=2 else all_cols)

if not selected_assets:
    st.error("Please select at least one asset.")
    st.stop()

# Dynamic Weight Inputs
st.sidebar.write(" **Assign Weights (%)**")
weights_input = {}
weight_cols = st.sidebar.columns(1) # Stack them vertically for clarity

total_weight = 0
for asset in selected_assets:
    # default to equal weight
    def_w = 100.0 / len(selected_assets)
    w = st.sidebar.number_input(f"{asset} %", min_value=0.0, max_value=100.0, value=float(int(def_w)), step=5.0)
    weights_input[asset] = w / 100.0
    total_weight += w

# Validation
if abs(total_weight - 100.0) > 0.01:
    st.sidebar.error(f"Total Weight: {total_weight:.2f}%. Must be 100%.")
else:
    st.sidebar.success(f"Total Weight: 100%")

# 3. Benchmark & Settings
st.sidebar.subheader("Settings")
# Try to find Nifty 50 as default
nifty_candidates = [c for c in all_cols if "Nifty" in c and "50" in c]
default_bench = nifty_candidates[0] if nifty_candidates else all_cols[0]

benchmark_col = st.sidebar.selectbox("Benchmark", all_cols, index=all_cols.index(default_bench))

rebalance_freq = st.sidebar.selectbox("Rebalancing", ["Daily", "Monthly", "Yearly", "Never (Buy & Hold)"], index=1)

# 4. Dates
valid_data = df_raw[selected_assets + [benchmark_col]].dropna()
if valid_data.empty:
    st.error("Selected assets have no overlapping data range.")
    st.stop()

min_d, max_d = valid_data.index.min().date(), valid_data.index.max().date()
default_start = max(min_d, date(2014, 1, 1))

start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=min_d, max_value=max_d)
end_date = st.sidebar.date_input("End Date", value=max_d, min_value=min_d, max_value=max_d)

if start_date >= end_date:
    st.error("Start Date must be before End Date")
    st.stop()

# --- MAIN EXECUTION ---

# Filter Data
mask = (df_raw.index.date >= start_date) & (df_raw.index.date <= end_date)
df_slice = df_raw.loc[mask].copy()

if abs(total_weight - 100.0) > 0.01:
    st.warning("Please correct weights in the sidebar to equal 100% to view results.")
    st.stop()

# Run Backtest
strat_series = run_portfolio_backtest(df_slice, weights_input, rebalance_freq)

if strat_series is None:
    st.error("Not enough data to calculate returns.")
    st.stop()

# Normalize Benchmark
bench_series = df_slice[benchmark_col].loc[strat_series.index]
bench_series = (bench_series / bench_series.iloc[0]) * 100

# Combined DF
res_df = pd.DataFrame({'Strategy': strat_series, 'Benchmark': bench_series})

# --- METRICS & VISUALS ---

st.title("Multi-Asset Strategy Dashboard")

# Top Metrics
years = (res_df.index[-1] - res_df.index[0]).days / 365.25

def calc_metrics(series):
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1
    ret = series.pct_change().dropna()
    vol = ret.std() * np.sqrt(252)
    return cagr, vol

s_cagr, s_vol = calc_metrics(res_df['Strategy'])
b_cagr, b_vol = calc_metrics(res_df['Benchmark'])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Strategy CAGR", f"{s_cagr:.2%}")
col2.metric("Benchmark CAGR", f"{b_cagr:.2%}", delta=f"{(s_cagr-b_cagr)*100:.2f} pts")
col3.metric("Strategy Volatility", f"{s_vol:.2%}")
col4.metric("Benchmark Volatility", f"{b_vol:.2%}", delta=f"{(s_vol-b_vol)*100:.2f} pts", delta_color="inverse")

# Main Chart
st.subheader("Growth of 100")
st.plotly_chart(px.line(res_df, title="Portfolio Performance"), use_container_width=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Yearly Returns", "Monthly Heatmap", "Drawdowns"])

with tab1:
    st.subheader("Calendar Year Returns")
    y_df = res_df.resample('YE').last()
    
    # Handle first year partial return
    # Add initial 100 row at start_date - 1 day for accurate first year pct_change
    start_row_date = res_df.index[0] - pd.Timedelta(days=1)
    start_row = pd.DataFrame([[100, 100]], columns=['Strategy', 'Benchmark'], index=[start_row_date])
    
    # Concatenate strictly for calculation
    calc_df = pd.concat([start_row, y_df]).sort_index()
    y_ret = calc_df.pct_change().dropna() # This drops the dummy start row, leaves real years
    
    y_ret['Alpha'] = y_ret['Strategy'] - y_ret['Benchmark']
    y_ret.index = y_ret.index.year
    
    st.dataframe(y_ret.style.format("{:.2%}").background_gradient(cmap='RdYlGn', subset=['Strategy', 'Alpha']), use_container_width=True)

with tab2:
    st.subheader("Strategy Monthly Returns")
    m_ret = res_df['Strategy'].resample('ME').apply(lambda x: (x.iloc[-1]/x.iloc[0]) - 1 if len(x) > 0 else 0)
    
    heat_df = pd.DataFrame({'Year': m_ret.index.year, 'Month': m_ret.index.strftime('%b'), 'Ret': m_ret.values})
    piv = heat_df.pivot(index='Year', columns='Month', values='Ret')
    piv = piv.reindex(columns=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    
    # Add YTD
    ytd = res_df['Strategy'].resample('YE').apply(lambda x: (x.iloc[-1]/x.iloc[0]) - 1)
    piv['YTD'] = ytd.values
    
    st.dataframe(piv.style.format("{:.2%}", na_rep="-").background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)

with tab3:
    st.subheader("Drawdowns")
    dd = (res_df / res_df.cummax()) - 1
    st.plotly_chart(px.area(dd, title="Underwater Plot"), use_container_width=True)
    st.write(f"**Max Drawdown (Strategy):** {dd['Strategy'].min():.2%}")