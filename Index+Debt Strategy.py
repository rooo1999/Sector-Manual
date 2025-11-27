import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import date

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Strategy Backtester", layout="wide")

# --- FILE LOADING LOGIC ---
DEFAULT_PATH = r"D:\MIRA Money\Data I've Analyzed\Individual Sheets\Index Data for Strategy.xlsx"

@st.cache_data
def load_data(uploaded_file=None):
    df = None
    
    # 1. Load File
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
            st.warning(f"Found file at default path but could not load: {e}")
            return None
    else:
        return None

    if df is not None:
        # 2. Clean Data
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Ensure Date is datetime
        # We assume the first column is Date
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]) # Drop rows where date is invalid
        
        df = df.rename(columns={date_col: 'Date'})
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
        
        # Forward fill to handle small data gaps (holidays), then drop rows that are still NaN
        df = df.ffill()
        
    return df

# --- BACKTEST ENGINE ---
def run_backtest(df_slice, equity_col, debt_col, equity_target, rebalance_freq):
    """
    Calculates portfolio NAV based on rebalancing frequency.
    """
    # Calculate daily returns for the individual assets
    returns = df_slice[[equity_col, debt_col]].pct_change().dropna()
    
    # Align df_slice to where returns start (lost first day due to pct_change)
    returns_idx = returns.index
    
    # Initialize Portfolio Data
    # We will simulate the value of an Equity Component and a Debt Component
    
    # Initial weights
    w_eq = equity_target
    w_debt = 1.0 - equity_target
    
    # These lists will store the daily value of the portfolio components
    # Start with 100 total value
    eq_vals = [100 * w_eq]
    debt_vals = [100 * w_debt]
    dates = [returns_idx[0]]
    
    # Pre-calculate integers for performance in loop
    n_days = len(returns)
    eq_ret_arr = returns[equity_col].values
    debt_ret_arr = returns[debt_col].values
    date_arr = returns.index
    
    # Logic Map
    is_daily = rebalance_freq == "Daily"
    is_never = rebalance_freq == "Never (Buy & Hold)"
    is_monthly = rebalance_freq == "Monthly"
    is_yearly = rebalance_freq == "Yearly"
    
    curr_eq = 100 * w_eq
    curr_debt = 100 * w_debt
    
    # Iterate from day 1 to end (Day 0 is already set)
    for i in range(1, n_days):
        today_date = date_arr[i]
        prev_date = date_arr[i-1]
        
        # 1. Apply Returns
        curr_eq *= (1 + eq_ret_arr[i])
        curr_debt *= (1 + debt_ret_arr[i])
        
        total_val = curr_eq + curr_debt
        
        # 2. Check Rebalance Trigger
        rebalance = False
        
        if is_daily:
            rebalance = True
        elif is_never:
            rebalance = False
        elif is_monthly:
            # Rebalance if month changed
            if today_date.month != prev_date.month:
                rebalance = True
        elif is_yearly:
            # Rebalance if year changed
            if today_date.year != prev_date.year:
                rebalance = True
                
        # 3. Execute Rebalance if needed
        if rebalance:
            curr_eq = total_val * w_eq
            curr_debt = total_val * w_debt
            
        eq_vals.append(curr_eq)
        debt_vals.append(curr_debt)
        dates.append(today_date)
        
    # Construct Result DataFrame
    res_df = pd.DataFrame({
        'Strategy': [e + d for e, d in zip(eq_vals, debt_vals)]
    }, index=dates)
    
    return res_df

# --- SIDEBAR & SETUP ---
st.sidebar.header("Configuration")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload Data (XLSX/CSV)", type=['xlsx', 'csv'])
df_raw = load_data(uploaded_file)

if df_raw is None:
    st.info(f"Using default path: {DEFAULT_PATH}")
    # Try loading default again explicitly if upload failed/empty
    if os.path.exists(DEFAULT_PATH):
        try:
            if DEFAULT_PATH.endswith('.csv'):
                df_raw = pd.read_csv(DEFAULT_PATH)
            else:
                df_raw = pd.read_excel(DEFAULT_PATH)
            
            # Quick Clean Repeat
            df_raw.columns = df_raw.columns.str.strip()
            df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0], errors='coerce')
            df_raw = df_raw.dropna(subset=[df_raw.columns[0]])
            df_raw = df_raw.rename(columns={df_raw.columns[0]: 'Date'}).set_index('Date').sort_index().ffill()
        except:
            st.error("Could not load default file. Please upload a file.")
            st.stop()
    else:
        st.error("Default file not found. Please upload.")
        st.stop()

# Asset Selection
cols = df_raw.columns.tolist()

# Smart Defaults
def find_col(keywords):
    for c in cols:
        if any(k.lower() in c.lower() for k in keywords):
            return c
    return None

eq_default = find_col(['Smallcap', '250', 'Midcap']) or cols[0]
debt_default = find_col(['Money', 'Liquid', 'Tata', 'Debt']) or (cols[1] if len(cols)>1 else cols[0])
bench_default = find_col(['Nifty', '50']) or cols[0]

st.sidebar.subheader("Asset Selection")
equity_col = st.sidebar.selectbox("Equity Component", cols, index=cols.index(eq_default))
debt_col = st.sidebar.selectbox("Debt Component", cols, index=cols.index(debt_default))
benchmark_col = st.sidebar.selectbox("Benchmark", cols, index=cols.index(bench_default))

# Parameters
st.sidebar.subheader("Strategy Parameters")
equity_weight = st.sidebar.slider("Equity Allocation (%)", 0, 100, 70, 5) / 100.0
rebalance_freq = st.sidebar.selectbox(
    "Rebalancing Frequency", 
    ["Daily", "Monthly", "Yearly", "Never (Buy & Hold)"],
    index=2
)

# Date Selection - Auto constrained to intersection of data
# Find range where both assets have data
valid_data = df_raw[[equity_col, debt_col]].dropna()
if valid_data.empty:
    st.error("Selected assets have no overlapping data. Please check input file.")
    st.stop()

min_d = valid_data.index.min().date()
max_d = valid_data.index.max().date()

st.sidebar.subheader("Backtest Period")
start_date = st.sidebar.date_input("Start Date", value=max(min_d, date(2014,1,1)), min_value=min_d, max_value=max_d)
end_date = st.sidebar.date_input("End Date", value=max_d, min_value=min_d, max_value=max_d)

if start_date >= end_date:
    st.error("Start Date must be before End Date")
    st.stop()

# --- PROCESSING ---

# Filter Data
mask = (df_raw.index.date >= start_date) & (df_raw.index.date <= end_date)
df_slice = df_raw.loc[mask].copy()

# 1. Calculate Benchmark NAV (Rebased to 100)
bench_series = df_slice[benchmark_col]
bench_nav = (bench_series / bench_series.iloc[0]) * 100

# 2. Run Strategy Backtest
strat_nav_df = run_backtest(df_slice, equity_col, debt_col, equity_weight, rebalance_freq)
strat_nav = strat_nav_df['Strategy']

# 3. Combine for metrics (Use common index)
common_idx = strat_nav.index.intersection(bench_nav.index)
strat_nav = strat_nav.loc[common_idx]
bench_nav = bench_nav.loc[common_idx]

# --- DASHBOARD LAYOUT ---

st.title("Strategy Backtest Dashboard")
st.markdown(f"""
**Allocation:** {int(equity_weight*100)}% {equity_col} | {int((1-equity_weight)*100)}% {debt_col}  
**Rebalancing:** {rebalance_freq}
""")

# METRICS
total_years = (strat_nav.index[-1] - strat_nav.index[0]).days / 365.25

def cagr(series):
    return (series.iloc[-1] / series.iloc[0]) ** (1/total_years) - 1

def vol(series):
    ret = series.pct_change().dropna()
    return ret.std() * np.sqrt(252)

s_cagr = cagr(strat_nav)
b_cagr = cagr(bench_nav)
s_vol = vol(strat_nav)
b_vol = vol(bench_nav)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Strategy CAGR", f"{s_cagr:.2%}")
c2.metric("Benchmark CAGR", f"{b_cagr:.2%}", delta=f"{(s_cagr-b_cagr)*100:.2f} pts")
c3.metric("Strategy Vol", f"{s_vol:.2%}")
c4.metric("Benchmark Vol", f"{b_vol:.2%}", delta=f"{(s_vol-b_vol)*100:.2f} pts", delta_color="inverse")

# CHART
chart_df = pd.DataFrame({'Strategy': strat_nav, 'Benchmark': bench_nav})
fig = px.line(chart_df, title="Portfolio Growth (Base 100)")
st.plotly_chart(fig, use_container_width=True)

# ANALYSIS TABS
t1, t2, t3 = st.tabs(["Yearly Returns", "Monthly Heatmap", "Drawdowns"])

# Helper for returns
strat_ret_daily = strat_nav.pct_change().dropna()

with t1:
    st.subheader("Calendar Year Returns")
    # Resample to Year End
    y_nav = chart_df.resample('YE').last()
    
    # Add start value to calculate first year return correctly
    start_row = pd.DataFrame([100, 100], index=['Strategy', 'Benchmark'], columns=[chart_df.index[0] - pd.Timedelta(days=1)]).T
    # Note: Simple pct_change on resampled data is usually sufficient for full years
    y_ret = y_nav.pct_change()
    
    # Handle first partial year if needed, but standard pct_change is cleaner for display
    # Let's verify if first year is NaN
    if pd.isna(y_ret.iloc[0,0]):
        # Calculate first year manually from start date
        first_yr_ret = (y_nav.iloc[0] / 100) - 1
        y_ret.iloc[0] = first_yr_ret
        
    y_ret['Alpha'] = y_ret['Strategy'] - y_ret['Benchmark']
    y_ret.index = y_ret.index.year
    
    st.dataframe(y_ret.style.format("{:.2%}")
                 .background_gradient(cmap='RdYlGn', subset=['Strategy', 'Alpha']), 
                 use_container_width=True)

with t2:
    st.subheader("Monthly Returns Heatmap (Strategy)")
    m_ret = strat_nav.resample('ME').apply(lambda x: (x.iloc[-1]/x.iloc[0]) - 1 if len(x)>0 else 0)
    
    heatmap = pd.DataFrame({
        'Year': m_ret.index.year,
        'Month': m_ret.index.strftime('%b'),
        'Return': m_ret.values
    })
    
    piv = heatmap.pivot(index='Year', columns='Month', values='Return')
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    piv = piv.reindex(columns=month_order)
    
    # Add YTD column
    ytd_s = strat_nav.resample('YE').apply(lambda x: (x.iloc[-1]/x.iloc[0]) - 1)
    piv['YTD'] = ytd_s.values
    
    st.dataframe(piv.style.format("{:.2%}", na_rep="-").background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)

with t3:
    st.subheader("Drawdown Analysis")
    def get_dd(series):
        roll_max = series.cummax()
        dd = (series - roll_max) / roll_max
        return dd
    
    dd_df = pd.DataFrame({
        'Strategy DD': get_dd(strat_nav),
        'Benchmark DD': get_dd(bench_nav)
    })
    
    fig_dd = px.area(dd_df, title="Underwater Plot")
    st.plotly_chart(fig_dd, use_container_width=True)
    
    max_dd_s = dd_df['Strategy DD'].min()
    max_dd_b = dd_df['Benchmark DD'].min()
    
    st.write(f"**Max Drawdown Strategy:** {max_dd_s:.2%}")
    st.write(f"**Max Drawdown Benchmark:** {max_dd_b:.2%}")