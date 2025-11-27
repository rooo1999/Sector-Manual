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
    
    # Priority 1: User uploaded a file
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            return None

    # Priority 2: Local default path
    elif os.path.exists(DEFAULT_PATH):
        try:
            if DEFAULT_PATH.endswith('.csv'):
                df = pd.read_csv(DEFAULT_PATH)
            else:
                df = pd.read_excel(DEFAULT_PATH)
        except Exception as e:
            st.warning(f"Found file at {DEFAULT_PATH} but could not load it: {e}")
            return None
    
    else:
        return None

    if df is not None:
        # Standardizing date column
        # Ensure the first column is treated as date
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.rename(columns={df.columns[0]: 'Date'})
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
        
        # Clean data: Forward fill missing data first, then drop remaining NaNs
        df = df.ffill().dropna()
        
    return df

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Strategy Configuration")

# 1. Data Loader
uploaded_file = st.sidebar.file_uploader("Upload Index Data (Optional)", type=['xlsx', 'csv'])
df_raw = load_data(uploaded_file)

if df_raw is None:
    st.info(f"Please upload the data file or ensure it exists at: {DEFAULT_PATH}")
    st.stop()

# 2. Asset Selection
st.sidebar.subheader("Asset Allocation")
available_cols = df_raw.columns.tolist()

# Defaults auto-detection
try:
    default_equity = [c for c in available_cols if "Smallcap" in c or "250" in c][0]
    default_debt = [c for c in available_cols if "Money" in c or "Tata" in c][0]
    default_benchmark = [c for c in available_cols if "Nifty" in c and "50" in c][0]
except:
    default_equity = available_cols[0]
    default_debt = available_cols[1] if len(available_cols) > 1 else available_cols[0]
    default_benchmark = available_cols[0]

equity_col = st.sidebar.selectbox("Select Equity Component", available_cols, index=available_cols.index(default_equity) if default_equity in available_cols else 0)
debt_col = st.sidebar.selectbox("Select Debt/Money Market Component", available_cols, index=available_cols.index(default_debt) if default_debt in available_cols else 1)
benchmark_col = st.sidebar.selectbox("Select Benchmark", available_cols, index=available_cols.index(default_benchmark) if default_benchmark in available_cols else 0)

# 3. Ratio Slider
equity_weight = st.sidebar.slider("Equity Allocation (%)", 0, 100, 70, 5) / 100.0
debt_weight = 1.0 - equity_weight

# 4. Date Selection
st.sidebar.subheader("Backtest Period")
min_date = df_raw.index.min().date()
max_date = df_raw.index.max().date()
default_start = date(2014, 1, 1)

if default_start < min_date:
    default_start = min_date

start_date = st.sidebar.date_input("Start Date", default_start, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

# --- CALCULATIONS ---

mask = (df_raw.index.date >= start_date) & (df_raw.index.date <= end_date)
df = df_raw.loc[mask].copy()

if df.empty:
    st.error("No data available for the selected date range.")
    st.stop()

# FIX: Specify fill_method=None to avoid FutureWarning
returns_df = df.pct_change(fill_method=None).dropna()

# Strategy Return
returns_df['Strategy'] = (returns_df[equity_col] * equity_weight) + (returns_df[debt_col] * debt_weight)

# Cumulative NAV
nav_df = (1 + returns_df).cumprod() * 100
nav_df.iloc[0] = 100 

# --- MAIN DASHBOARD ---

st.title("Strategy Backtest Dashboard")
st.markdown(f"**Strategy:** {int(equity_weight*100)}% {equity_col} + {int(debt_weight*100)}% {debt_col}")

# 1. Summary Metrics
total_years = (df.index[-1] - df.index[0]).days / 365.25

def get_cagr(series):
    if total_years <= 0: return 0
    return (series.iloc[-1] / series.iloc[0]) ** (1/total_years) - 1

def get_volatility(series):
    return series.std() * np.sqrt(252)

strat_cagr = get_cagr(nav_df['Strategy'])
bench_cagr = get_cagr(nav_df[benchmark_col])
strat_vol = get_volatility(returns_df['Strategy'])
bench_vol = get_volatility(returns_df[benchmark_col])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Strategy CAGR", f"{strat_cagr:.2%}")
col2.metric("Benchmark CAGR", f"{bench_cagr:.2%}", delta=f"{(strat_cagr-bench_cagr)*100:.2f} pts")
col3.metric("Strategy Volatility", f"{strat_vol:.2%}")
col4.metric("Benchmark Volatility", f"{bench_vol:.2%}", delta=f"{(strat_vol-bench_vol)*100:.2f} pts", delta_color="inverse")

# 2. Performance Chart
st.subheader("Performance Comparison (Rebased to 100)")
fig = px.line(nav_df, y=['Strategy', benchmark_col], title="Growth of 100")
st.plotly_chart(fig, use_container_width=True)

# --- DETAILED ANALYSIS ---

tab1, tab2 = st.tabs(["Yearly Analysis", "Monthly Heatmap"])

with tab1:
    st.subheader("Yearly Returns Comparison")
    
    # Resample to Yearly
    yearly_nav = nav_df.resample('YE').last()
    
    # FIX: Handle pct_change deprecation
    yearly_returns = yearly_nav.pct_change(fill_method=None)
    
    display_yearly = yearly_returns[['Strategy', benchmark_col]].copy()
    display_yearly['Alpha'] = display_yearly['Strategy'] - display_yearly[benchmark_col]
    
    # Drop the first row if it's NaN (common in first year calc)
    display_yearly = display_yearly.dropna()

    st.dataframe(display_yearly.style.format("{:.2%}")
                 .background_gradient(cmap='RdYlGn', subset=['Strategy', 'Alpha']), 
                 use_container_width=True)

with tab2:
    st.subheader("Monthly Returns (Strategy)")
    
    monthly_ret = returns_df['Strategy'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
    monthly_ret_df = pd.DataFrame(monthly_ret)
    monthly_ret_df['Year'] = monthly_ret_df.index.year
    monthly_ret_df['Month'] = monthly_ret_df.index.strftime('%b')
    
    pivot_table = monthly_ret_df.pivot(index='Year', columns='Month', values='Strategy')
    
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_table = pivot_table.reindex(columns=months_order)
    
    ytd = returns_df['Strategy'].resample('YE').apply(lambda x: (1 + x).prod() - 1)
    pivot_table['YTD'] = ytd.values
    
    st.dataframe(pivot_table.style.format("{:.2%}", na_rep="-")
                 .background_gradient(cmap='RdYlGn', axis=None), 
                 use_container_width=True)

# 3. Risk Metrics
st.subheader("Risk Metrics Breakdown")
risk_df = pd.DataFrame(index=['Strategy', benchmark_col])
risk_df['Annualized Volatility'] = [strat_vol, bench_vol]
risk_df['Sharpe Ratio (Rf=0%)'] = [strat_cagr/strat_vol if strat_vol > 0 else 0, bench_cagr/bench_vol if bench_vol > 0 else 0]

def calculate_max_drawdown(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

risk_df['Max Drawdown'] = [calculate_max_drawdown(nav_df['Strategy']), calculate_max_drawdown(nav_df[benchmark_col])]

st.table(risk_df.style.format("{:.2%}"))