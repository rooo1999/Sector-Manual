import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import date

st.set_page_config(page_title="Advanced Basket Strategy", layout="wide")

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

# --- CALCULATIONS ---

def run_fixed_allocation(df_slice, weights_dict, rebalance_freq):
    assets = list(weights_dict.keys())
    returns = df_slice[assets].pct_change().dropna()
    if returns.empty: return None
    
    vals = np.array([100.0 * weights_dict[a] for a in assets])
    hist, dates = [100.0], [returns.index[0]]
    ret_arr = returns.values
    target_w = np.array([weights_dict[a] for a in assets])
    
    is_daily = rebalance_freq == "Daily"
    is_monthly = rebalance_freq == "Monthly"
    is_yearly = rebalance_freq == "Yearly"
    idx = returns.index
    
    for i in range(1, len(idx)):
        vals = vals * (1 + ret_arr[i])
        total = np.sum(vals)
        
        reb = False
        if is_daily: reb = True
        elif is_monthly and idx[i].month != idx[i-1].month: reb = True
        elif is_yearly and idx[i].year != idx[i-1].year: reb = True
        
        if reb and "Never" not in rebalance_freq:
            vals = total * target_w
            
        hist.append(total)
        dates.append(idx[i])
        
    return pd.Series(hist, index=dates, name="Strategy")

def run_basket_trend(df_slice, risky_weights, safe_asset, ma_period):
    """
    1. Construct Synthetic Risky Asset (Daily Rebalanced Basket)
    2. Calc MA on Synthetic
    3. Switch between Synthetic Return and Safe Return
    """
    risky_assets = list(risky_weights.keys())
    
    # Get Returns
    all_assets = risky_assets + [safe_asset]
    returns = df_slice[all_assets].pct_change().dropna()
    
    if returns.empty: return None
    
    # 1. Create Synthetic Risky Return Series (Daily Rebalanced)
    # This represents how your "Equity Basket" performs day-to-day
    risky_daily_ret = pd.Series(0.0, index=returns.index)
    for asset, weight in risky_weights.items():
        risky_daily_ret += returns[asset] * weight
        
    # 2. Create Synthetic NAV (to calculate MA)
    synthetic_nav = (1 + risky_daily_ret).cumprod() * 100
    synthetic_ma = synthetic_nav.rolling(window=ma_period).mean()
    
    # 3. Generate Signal
    # Signal = Previous Day NAV > Previous Day MA
    # We shift by 1 to avoid lookahead bias (decision made on yesterday's close)
    signal = (synthetic_nav > synthetic_ma).shift(1)
    
    # 4. Apply Strategy
    # If Signal True (Bullish) -> Get Risky Basket Return
    # If Signal False (Bearish) -> Get Safe Asset Return
    # If Signal NaN (Start) -> Get Safe Asset Return
    
    safe_ret = returns[safe_asset]
    
    # Vectorized condition
    strategy_daily_ret = np.where(signal == True, risky_daily_ret, safe_ret)
    
    # Handle NaNs at start (due to MA window) by defaulting to 0 or safe
    strategy_daily_ret = np.nan_to_num(strategy_daily_ret, nan=0.0)
    
    # Calculate Final Strategy NAV
    strat_nav = (1 + strategy_daily_ret).cumprod() * 100
    
    return pd.Series(strat_nav, index=returns.index, name="Strategy")

# --- UI SETUP ---
st.sidebar.header("Data Configuration")
f = st.sidebar.file_uploader("Upload File", type=['xlsx','csv'])
df = load_data(f)

if df is None: 
    st.info("Please upload data.")
    st.stop()

cols = df.columns.tolist()

# --- STRATEGY UI ---
st.sidebar.header("Strategy Logic")
mode = st.sidebar.selectbox("Mode", ["Fixed Allocation (Passive)", "Trend Following (Active Switch)"])

# Common Weights Input Function
def get_weights(label_suffix=""):
    st.sidebar.markdown(f"**Select {label_suffix} Assets:**")
    assets = st.sidebar.multiselect(f"Assets {label_suffix}", cols, key=f"ms_{label_suffix}")
    w = {}
    if not assets: return None
    
    total = 0
    st.sidebar.markdown("Define Weights:")
    c = st.sidebar.columns(1)
    for a in assets:
        val = st.sidebar.number_input(f"{a} %", 0, 100, int(100/len(assets)), key=f"num_{a}_{label_suffix}")
        w[a] = val/100.0
        total += val
        
    if total != 100:
        st.sidebar.error(f"Total {total}%. Must be 100%.")
        return None
    return w

# Mode Specific Inputs
weights_input = None
safe_asset = None
ma_days = 200

if mode == "Fixed Allocation (Passive)":
    weights_input = get_weights("(Portfolio)")
    freq = st.sidebar.selectbox("Rebalancing", ["Daily", "Monthly", "Yearly", "Never"])
else:
    st.sidebar.info("Construct your 'Risky Basket'. If this basket drops below MA, we switch to Safe Asset.")
    risky_weights = get_weights("(Risky Basket)")
    
    st.sidebar.markdown("**Select Safe Asset:**")
    # Smart default for debt
    debt_candidates = [c for c in cols if "Money" in c or "Liquid" in c or "Debt" in c]
    def_debt = debt_candidates[0] if debt_candidates else cols[0]
    safe_asset = st.sidebar.selectbox("Safe Asset (Cash/Debt)", cols, index=cols.index(def_debt))
    
    ma_days = st.sidebar.number_input("MA Days", value=200)

# Benchmark
bench = st.sidebar.selectbox("Benchmark", cols, index=0)

# Dates
min_d, max_d = df.index.min().date(), df.index.max().date()
s_date = st.sidebar.date_input("Start", max(min_d, date(2014,1,1)))
e_date = st.sidebar.date_input("End", max_d)

# --- EXECUTION ---
mask = (df.index.date >= s_date) & (df.index.date <= e_date)
df_slice = df.loc[mask]

if df_slice.empty: st.stop()

if mode == "Fixed Allocation (Passive)":
    if not weights_input: st.stop()
    res_series = run_fixed_allocation(df_slice, weights_input, freq)
else:
    if not risky_weights: st.stop()
    res_series = run_basket_trend(df_slice, risky_weights, safe_asset, ma_days)

if res_series is None: 
    st.error("Insufficient Data")
    st.stop()

# Align Bench
b_series = df_slice[bench].loc[res_series.index]
b_series = (b_series / b_series.iloc[0]) * 100

final = pd.DataFrame({'Strategy': res_series, 'Benchmark': b_series})

# --- METRICS & CHARTS ---
st.title(f"{mode} Backtest")

years = (final.index[-1] - final.index[0]).days / 365.25
cagr = lambda s: (s.iloc[-1]/s.iloc[0])**(1/years)-1
vol = lambda s: s.pct_change().std() * np.sqrt(252)
dd = lambda s: (s/s.cummax()-1).min()

s_c, b_c = cagr(final['Strategy']), cagr(final['Benchmark'])
s_v, b_v = vol(final['Strategy']), vol(final['Benchmark'])
s_d, b_d = dd(final['Strategy']), dd(final['Benchmark'])

m1, m2, m3, m4 = st.columns(4)
m1.metric("CAGR", f"{s_c:.2%}", f"{(s_c-b_c)*100:.2f} pts")
m2.metric("Vol", f"{s_v:.2%}", f"{(s_v-b_v)*100:.2f} pts", delta_color="inverse")
m3.metric("Max Drawdown", f"{s_d:.2%}")
m4.metric("Sharpe", f"{s_c/s_v:.2f}")

st.plotly_chart(px.line(final, title="Performance"), use_container_width=True)

# TABS
t1, t2 = st.tabs(["Yearly", "Monthly Heatmap"])
with t1:
    y = final.resample('YE').last().pct_change()
    # Fix first year
    f_val = final.resample('YE').last().iloc[0]
    y.iloc[0] = (f_val/100) - 1
    y['Alpha'] = y['Strategy'] - y['Benchmark']
    y.index = y.index.year
    st.dataframe(y.style.format("{:.2%}").background_gradient(cmap='RdYlGn', subset=['Strategy','Alpha']), use_container_width=True)

with t2:
    m = final['Strategy'].resample('ME').apply(lambda x: x.iloc[-1]/x.iloc[0]-1 if len(x)>0 else 0)
    p = pd.DataFrame({'Y':m.index.year, 'M':m.index.strftime('%b'), 'V':m.values}).pivot(index='Y', columns='M', values='V')
    p = p.reindex(columns=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    p['YTD'] = final['Strategy'].resample('YE').apply(lambda x: x.iloc[-1]/x.iloc[0]-1).values
    st.dataframe(p.style.format("{:.2%}", na_rep="-").background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)