import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import date

st.set_page_config(page_title="Pro Strategy Engine", layout="wide")

# --- DATA LOADER ---
DEFAULT_PATH = r"D:\MIRA Money\Data I've Analyzed\Individual Sheets\Index Data for Strategy.xlsx"

@st.cache_data
def load_data(uploaded_file=None):
    df = None
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
        except: return None
    elif os.path.exists(DEFAULT_PATH):
        try:
            if DEFAULT_PATH.endswith('.csv'): df = pd.read_csv(DEFAULT_PATH)
            else: df = pd.read_excel(DEFAULT_PATH)
        except: return None
    else: return None

    if df is not None:
        df.columns = df.columns.str.strip()
        df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], errors='coerce')
        df = df.dropna(subset=[df.columns[0]]).rename(columns={df.columns[0]: 'Date'}).sort_values('Date').set_index('Date')
        df = df.ffill().dropna()
    return df

# --- CORE LOGIC ---

def construct_basket(df_full, weights_dict, rebalance_freq):
    """
    Creates a synthetic index (NAV) for a basket of assets
    starting from the beginning of the file (to maximize history).
    """
    assets = list(weights_dict.keys())
    returns = df_full[assets].pct_change().fillna(0)
    
    vals = np.array([100.0 * weights_dict[a] for a in assets])
    hist = [100.0]
    dates = [returns.index[0]]
    
    ret_arr = returns.values
    target_w = np.array([weights_dict[a] for a in assets])
    idx = returns.index
    
    # Fast Loop
    for i in range(1, len(idx)):
        vals = vals * (1 + ret_arr[i])
        total = np.sum(vals)
        
        reb = False
        if rebalance_freq == "Daily": reb = True
        elif rebalance_freq == "Monthly" and idx[i].month != idx[i-1].month: reb = True
        elif rebalance_freq == "Yearly" and idx[i].year != idx[i-1].year: reb = True
        
        if reb and "Never" not in rebalance_freq:
            vals = total * target_w
            
        hist.append(total)
        dates.append(idx[i])
        
    return pd.Series(hist, index=dates, name="Basket_NAV")

def run_strategy_engine(df_full, mode, risky_weights, safe_asset, ma_days, rebal_freq, start_date, end_date):
    """
    1. Generates Risky Basket from full history (to prep MA).
    2. Calculates MA on full history.
    3. Slices to user date range.
    4. Executes Strategy.
    """
    # 1. Construct Risky Basket (The "Growth Engine")
    risky_nav = construct_basket(df_full, risky_weights, rebal_freq)
    
    # 2. Calculate Signals on Full Data (Solves Cold Start)
    ma_series = risky_nav.rolling(window=ma_days).mean()
    
    # Signal: If NAV > MA = True (Equity). Else False (Debt).
    # Shift(1) to avoid lookahead bias.
    raw_signal = (risky_nav > ma_series).shift(1)
    
    # 3. Slice Data to User Selection
    mask = (df_full.index.date >= start_date) & (df_full.index.date <= end_date)
    df_slice = df_full.loc[mask]
    
    # Also slice the pre-calculated series
    risky_nav = risky_nav.loc[mask]
    raw_signal = raw_signal.loc[mask]
    
    if df_slice.empty: return None, None

    # Get Returns for the specific period
    risky_ret = risky_nav.pct_change().fillna(0)
    safe_ret = df_slice[safe_asset].pct_change().fillna(0)
    
    # 4. Apply Strategy Logic
    if mode == "Fixed Allocation":
        # Just return the Risky Basket (which is already weighted)
        # Note: If fixed allocation includes Debt in the basket, it's already handled in construct_basket
        final_ret = risky_ret 
    else:
        # Trend Following: Switch between Risky Basket and Safe Asset
        # np.where(condition, if_true, if_false)
        final_ret = np.where(raw_signal == True, risky_ret, safe_ret)
        
    # Calculate Strategy NAV
    strat_nav = (1 + final_ret).cumprod() * 100
    strat_series = pd.Series(strat_nav, index=df_slice.index, name="Strategy")
    
    return strat_series, df_slice

# --- UI SETUP ---
st.sidebar.header("Data")
f = st.sidebar.file_uploader("Upload Data", type=['xlsx','csv'])
df_raw = load_data(f)
if df_raw is None: 
    st.info("Upload file to proceed.")
    st.stop()

cols = df_raw.columns.tolist()

# --- CONFIGURATION ---
st.sidebar.header("Strategy Settings")
mode = st.sidebar.selectbox("Strategy Mode", ["Trend Following (Dynamic)", "Fixed Allocation (Static)"])

# Helper for weights
def get_weights(label):
    st.sidebar.markdown(f"**{label}**")
    assets = st.sidebar.multiselect(f"Select Assets", cols, key=label)
    w_dict = {}
    if assets:
        cols_ui = st.sidebar.columns(1)
        total = 0
        for a in assets:
            val = st.sidebar.number_input(f"{a} %", 0, 100, int(100/len(assets)), key=f"w_{a}_{label}")
            w_dict[a] = val/100.0
            total += val
        if total != 100: st.sidebar.error("Weights must sum to 100%")
    return w_dict

risky_weights = {}
safe_asset = cols[0]
ma_days = 200
rebal_freq = "Monthly"

if mode == "Fixed Allocation (Static)":
    st.sidebar.info("Construct a fixed portfolio.")
    risky_weights = get_weights("Portfolio Composition")
    rebal_freq = st.sidebar.selectbox("Rebalancing Frequency", ["Daily","Monthly","Yearly","Never"])
else:
    st.sidebar.info("1. Construct Risky Basket (Equity). \n2. Select Safe Asset (Debt). \n3. Logic: If Basket < MA, switch to Debt.")
    risky_weights = get_weights("Step 1: Risky Basket (e.g. Mid + Small)")
    rebal_freq = st.sidebar.selectbox("Risky Basket Rebalancing", ["Daily","Monthly","Yearly","Never"], index=1)
    
    st.sidebar.markdown("**Step 2: Safe Asset**")
    # Auto-find debt
    d_guess = [c for c in cols if "Money" in c or "Liquid" in c or "Debt" in c]
    safe_asset = st.sidebar.selectbox("Defensive Fund", cols, index=cols.index(d_guess[0]) if d_guess else 0)
    
    ma_days = st.sidebar.number_input("Moving Average (DMA)", value=200, help="Trend period. Lower = Faster switching.")

# Benchmark
bench_col = st.sidebar.selectbox("Benchmark", cols, index=0)

# Dates
valid_dates = df_raw.index
s_date = st.sidebar.date_input("Start Date", max(valid_dates.min().date(), date(2014,1,1)))
e_date = st.sidebar.date_input("End Date", valid_dates.max().date())

# --- EXECUTION ---
if not risky_weights or sum(risky_weights.values()) != 1.0:
    st.warning("Please configure assets and ensure weights sum to 100%.")
    st.stop()

strat_series, df_sliced = run_strategy_engine(df_raw, mode, risky_weights, safe_asset, ma_days, rebal_freq, s_date, e_date)

if strat_series is None:
    st.error("Error running strategy. Check dates.")
    st.stop()

# Bench Setup
bench_series = df_sliced[bench_col]
bench_series = (bench_series / bench_series.iloc[0]) * 100

# Combine
final_df = pd.DataFrame({'Strategy': strat_series, 'Benchmark': bench_series})

# --- DASHBOARD METRICS ---
st.title("Strategy Analytics Dashboard")

# 1. Calculator
def get_stats(series):
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    years = (series.index[-1] - series.index[0]).days / 365.25
    
    cagr = (end_val/start_val)**(1/years) - 1
    
    # Volatility
    daily_ret = series.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)
    
    # Drawdown
    roll_max = series.cummax()
    dd = (series - roll_max) / roll_max
    max_dd = dd.min()
    
    # Sharpe (Rf=6% assumption for Indian context)
    rf = 0.06
    sharpe = (cagr - rf) / vol
    
    return cagr, vol, max_dd, sharpe, end_val

s_cagr, s_vol, s_dd, s_sharpe, s_end = get_stats(final_df['Strategy'])
b_cagr, b_vol, b_dd, b_sharpe, b_end = get_stats(final_df['Benchmark'])

# 2. Display Top Cards
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("CAGR", f"{s_cagr:.2%}", f"{(s_cagr-b_cagr)*100:.2f} pts")
k2.metric("Max Drawdown", f"{s_dd:.2%}", f"{(s_dd-b_dd)*100:.2f} pts", delta_color="inverse")
k3.metric("Volatility", f"{s_vol:.2%}", f"{(s_vol-b_vol)*100:.2f} pts", delta_color="inverse")
k4.metric("Sharpe Ratio", f"{s_sharpe:.2f}", f"{s_sharpe-b_sharpe:.2f}")
k5.metric("Final Value (of 100)", f"{s_end:.0f}", f"{s_end-b_end:.0f}")

# 3. Main Chart
st.subheader("Growth of ₹100")
fig = px.line(final_df, title="")
st.plotly_chart(fig, use_container_width=True)

# 4. Detailed Stats Table
st.subheader("Detailed Performance Comparison")
stats_data = {
    "Metric": ["CAGR (Annual Return)", "Volatility (Risk)", "Max Drawdown", "Sharpe Ratio", "Total Return"],
    "Strategy": [f"{s_cagr:.2%}", f"{s_vol:.2%}", f"{s_dd:.2%}", f"{s_sharpe:.2f}", f"{(s_end/100 - 1):.2%}"],
    "Benchmark": [f"{b_cagr:.2%}", f"{b_vol:.2%}", f"{b_dd:.2%}", f"{b_sharpe:.2f}", f"{(b_end/100 - 1):.2%}"],
    "Difference": [f"{(s_cagr-b_cagr)*100:.2f} pts", f"{(s_vol-b_vol)*100:.2f} pts", f"{(s_dd-b_dd)*100:.2f} pts", f"{s_sharpe-b_sharpe:.2f}", "-"]
}
st.table(pd.DataFrame(stats_data))

# 5. Drawdown Chart
st.subheader("Drawdown Analysis (Risk)")
dd_df = (final_df / final_df.cummax()) - 1
fig_dd = px.area(dd_df, title="Underwater Plot (Depth of Losses)")
st.plotly_chart(fig_dd, use_container_width=True)

# 6. Returns Heatmap
t1, t2 = st.tabs(["Strategy Heatmap", "Yearly Comparison"])

with t1:
    strat_ret = final_df['Strategy'].resample('ME').apply(lambda x: x.iloc[-1]/x.iloc[0]-1 if len(x)>0 else 0)
    piv = pd.DataFrame({'Y': strat_ret.index.year, 'M': strat_ret.index.strftime('%b'), 'V': strat_ret.values})
    piv = piv.pivot(index='Y', columns='M', values='V')
    piv = piv.reindex(columns=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    piv['YTD'] = final_df['Strategy'].resample('YE').apply(lambda x: x.iloc[-1]/x.iloc[0]-1).values
    st.dataframe(piv.style.format("{:.2%}", na_rep="-").background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)

with t2:
    # Accurate Yearly calc including partial years
    y_df = final_df.resample('YE').last()
    # Add start row
    start_row = pd.DataFrame([100, 100], index=['Strategy','Benchmark'], columns=[final_df.index[0]-pd.Timedelta(days=1)]).T
    calc_df = pd.concat([start_row, y_df]).sort_index()
    y_ret = calc_df.pct_change().dropna()
    
    y_ret['Alpha'] = y_ret['Strategy'] - y_ret['Benchmark']
    y_ret.index = y_ret.index.year
    st.dataframe(y_ret.style.format("{:.2%}").background_gradient(cmap='RdYlGn', subset=['Strategy', 'Alpha']), use_container_width=True)