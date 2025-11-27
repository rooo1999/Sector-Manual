import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import date

st.set_page_config(page_title="Advanced Strategy Backtester", layout="wide")

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

# --- STRATEGY ENGINES ---

def run_fixed_allocation(df_slice, weights_dict, rebalance_freq):
    assets = list(weights_dict.keys())
    returns = df_slice[assets].pct_change().dropna()
    if returns.empty: return None
    
    idx = returns.index
    n = len(idx)
    vals = np.array([100.0 * weights_dict[a] for a in assets])
    hist, dates = [100.0], [idx[0]]
    
    ret_arr = returns.values
    target_w = np.array([weights_dict[a] for a in assets])
    
    is_daily = rebalance_freq == "Daily"
    is_monthly = rebalance_freq == "Monthly"
    is_yearly = rebalance_freq == "Yearly"
    
    for i in range(1, n):
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

def run_trend_following(df_slice, risky_asset, safe_asset, ma_period=200):
    """
    If Risky Asset > Moving Average: 100% Risky
    Else: 100% Safe
    """
    # Calculate MA
    prices = df_slice[risky_asset]
    ma_series = prices.rolling(window=ma_period).mean()
    
    # Combined DF
    data = pd.DataFrame({
        'Risky': df_slice[risky_asset],
        'Safe': df_slice[safe_asset],
        'MA': ma_series
    }).dropna()
    
    if data.empty: return None
    
    # Calculate returns
    risky_ret = data['Risky'].pct_change()
    safe_ret = data['Safe'].pct_change()
    
    # Signal (Shifted by 1 day to avoid lookahead bias)
    # If today Close > MA, tomorrow we are in Equity
    signal = (data['Risky'] > data['MA']).shift(1)
    
    # Logic
    # If Signal is True (1) -> Use Risky Ret, Else -> Use Safe Ret
    # If signal is NaN (first row), assume Safe
    strat_ret = np.where(signal == True, risky_ret, safe_ret)
    
    # Calculate NAV
    # Replace NaN in first row with 0
    strat_ret = np.nan_to_num(strat_ret, nan=0.0)
    
    nav = (1 + strat_ret).cumprod() * 100
    return pd.Series(nav, index=data.index, name="Strategy")

# --- UI ---
st.sidebar.header("Data")
f = st.sidebar.file_uploader("Upload", type=['xlsx','csv'])
df = load_data(f)
if df is None: st.stop()

cols = df.columns.tolist()

# --- STRATEGY SELECTION ---
st.sidebar.header("Strategy Settings")
strat_type = st.sidebar.selectbox("Strategy Type", ["Fixed Allocation (Buy & Hold)", "Trend Following (Timing)"])

if strat_type == "Fixed Allocation (Buy & Hold)":
    sel_assets = st.sidebar.multiselect("Select Assets", cols, default=cols[:2])
    weights = {}
    if not sel_assets: st.stop()
    for a in sel_assets:
        weights[a] = st.sidebar.number_input(f"{a} %", 0, 100, int(100/len(sel_assets))) / 100.0
    freq = st.sidebar.selectbox("Rebalance", ["Daily","Monthly","Yearly","Never"])
    
else:
    st.sidebar.info("Switch to Debt when Market crashes.")
    risky = st.sidebar.selectbox("Risky Asset (Equity)", cols, index=0)
    safe = st.sidebar.selectbox("Safe Asset (Debt/Liquid)", cols, index=1)
    ma_days = st.sidebar.number_input("Moving Average Days", value=200)

# --- BENCHMARK ---
bench = st.sidebar.selectbox("Benchmark", cols, index=0)

# --- DATES ---
min_d, max_d = df.index.min().date(), df.index.max().date()
s_date = st.sidebar.date_input("Start", max(min_d, date(2014,1,1)))
e_date = st.sidebar.date_input("End", max_d)

# --- RUN ---
mask = (df.index.date >= s_date) & (df.index.date <= e_date)
df_s = df.loc[mask]

if strat_type == "Fixed Allocation (Buy & Hold)":
    if sum(weights.values()) != 1.0: st.error("Weights != 100%"); st.stop()
    res = run_fixed_allocation(df_s, weights, freq)
else:
    res = run_trend_following(df_s, risky, safe, ma_days)

if res is None: st.error("Not enough data"); st.stop()

# Align Bench
b_series = df_s[bench].loc[res.index]
b_series = (b_series / b_series.iloc[0]) * 100
final = pd.DataFrame({'Strategy': res, 'Benchmark': b_series})

# --- DISPLAY ---
st.title(f"{strat_type} Results")

# Metrics
years = (final.index[-1] - final.index[0]).days / 365.25
cagr = lambda s: (s.iloc[-1]/s.iloc[0])**(1/years)-1
vol = lambda s: s.pct_change().std() * np.sqrt(252)

sc, bc = cagr(final['Strategy']), cagr(final['Benchmark'])
sv, bv = vol(final['Strategy']), vol(final['Benchmark'])

c1,c2,c3,c4 = st.columns(4)
c1.metric("Strategy CAGR", f"{sc:.2%}")
c2.metric("Benchmark CAGR", f"{bc:.2%}", delta=f"{(sc-bc)*100:.2f}")
c3.metric("Strategy Vol", f"{sv:.2%}")
c4.metric("Benchmark Vol", f"{bv:.2%}", delta=f"{(sv-bv)*100:.2f}", delta_color="inverse")

st.plotly_chart(px.line(final, title="Performance"), use_container_width=True)

t1, t2 = st.tabs(["Yearly Returns", "Monthly Heatmap"])
with t1:
    y = final.resample('YE').last().pct_change()
    # Fix first year
    first_year = final.index[0].year
    first_val = final.resample('YE').last().iloc[0]
    y.iloc[0] = (first_val / 100) - 1
    
    y['Alpha'] = y['Strategy'] - y['Benchmark']
    y.index = y.index.year
    st.dataframe(y.style.format("{:.2%}").background_gradient(cmap='RdYlGn', subset=['Strategy','Alpha']), use_container_width=True)

with t2:
    m = final['Strategy'].resample('ME').apply(lambda x: x.iloc[-1]/x.iloc[0]-1 if len(x)>0 else 0)
    piv = pd.DataFrame({'Y':m.index.year, 'M':m.index.strftime('%b'), 'V':m.values}).pivot(index='Y', columns='M', values='V')
    piv = piv.reindex(columns=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    piv['YTD'] = final['Strategy'].resample('YE').apply(lambda x: x.iloc[-1]/x.iloc[0]-1).values
    st.dataframe(piv.style.format("{:.2%}", na_rep="-").background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)