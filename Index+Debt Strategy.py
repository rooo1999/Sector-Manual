import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import date

st.set_page_config(page_title="200 EMA Strategy Engine", layout="wide")

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
        df = df.dropna(subset=[df.columns[0]])
        df = df.rename(columns={df.columns[0]: 'Date'}).sort_values('Date').set_index('Date')
        df = df.ffill().dropna()
    return df

# --- CORE LOGIC ---

def get_basket_nav(df, weights):
    assets = list(weights.keys())
    returns = df[assets].pct_change().fillna(0)
    
    vals = np.array([100.0 * weights[a] for a in assets])
    hist = [100.0]
    dates = [returns.index[0]]
    ret_arr = returns.values
    target_w = np.array([weights[a] for a in assets])
    idx = returns.index
    
    # Monthly Rebalancing
    for i in range(1, len(idx)):
        vals = vals * (1 + ret_arr[i])
        total = np.sum(vals)
        if idx[i].month != idx[i-1].month: 
            vals = total * target_w
        hist.append(total)
        dates.append(idx[i])
        
    return pd.Series(hist, index=dates, name="Risky_Basket")

def calculate_ma(series, window, ma_type):
    """
    Calculates SMA or EMA based on user selection.
    """
    if ma_type == "EMA (Exponential)":
        # EMA places more weight on recent data, reacting faster
        return series.ewm(span=window, adjust=False).mean()
    else:
        # SMA is the standard average
        return series.rolling(window=window, min_periods=1).mean()

def run_strategy(df_full, risky_weights, safe_asset, buffer_pct, start_date, end_date, ma_type, ma_window):
    
    # 1. Build Basket
    risky_nav = get_basket_nav(df_full, risky_weights)
    
    # 2. Calculate Moving Average (EMA/SMA)
    ma_series = calculate_ma(risky_nav, ma_window, ma_type)
    
    # 3. Apply Buffer Logic
    price = risky_nav.values
    ma = ma_series.values
    
    signals = []
    state = 1 # Start Invested
    
    buffer_mult = buffer_pct / 100.0
    
    for i in range(len(price)):
        p = price[i]
        m = ma[i]
        
        # Upper/Lower Bands
        upper = m * (1 + buffer_mult)
        lower = m * (1 - buffer_mult)
        
        if state == 1:
            # Exit Rule: Price < Lower Band
            if p < lower:
                state = 0
        else:
            # Entry Rule: Price > Upper Band
            if p > upper:
                state = 1
                
        signals.append(bool(state))
        
    raw_signal = pd.Series(signals, index=risky_nav.index)
    
    # 4. Shift Signal (Trade Next Day)
    trade_signal = raw_signal.shift(1).fillna(True)
    
    # 5. Slice
    mask = (df_full.index.date >= start_date) & (df_full.index.date <= end_date)
    df_slice = df_full.loc[mask]
    
    if df_slice.empty: return None, None, None, None, None

    risky_nav = risky_nav.loc[mask]
    trade_signal = trade_signal.loc[mask]
    ma_series = ma_series.loc[mask]
    
    # Rebase for visualization
    risky_nav_viz = (risky_nav / risky_nav.iloc[0]) * 100
    
    # 6. Returns
    risky_ret = risky_nav.pct_change().fillna(0)
    safe_ret = df_slice[safe_asset].pct_change().fillna(0)
    
    final_ret = np.where(trade_signal, risky_ret, safe_ret)
    strat_nav = (1 + final_ret).cumprod() * 100
    strat_series = pd.Series(strat_nav, index=df_slice.index, name="Strategy")
    
    return strat_series, risky_nav_viz, ma_series, trade_signal, df_slice

def analyze_trades(signal_series, strategy_nav):
    # Detect flips
    trades = signal_series.diff().fillna(0)
    
    # Entry: 0 -> 1 (Value +1)
    entries = trades[trades == 1].index
    # Exit: 1 -> 0 (Value -1)
    exits = trades[trades == -1].index
    
    # Handle initial state
    if signal_series.iloc[0] == True:
        entries = entries.insert(0, signal_series.index[0])
    
    # Handle open position at end
    if signal_series.iloc[-1] == True:
        exits = exits.append(pd.Index([signal_series.index[-1]]))
        
    # Zip
    n = min(len(entries), len(exits))
    entries = entries[:n]
    exits = exits[:n]
    
    log = []
    for en, ex in zip(entries, exits):
        try:
            val_in = strategy_nav.loc[en]
            val_out = strategy_nav.loc[ex]
            ret = (val_out/val_in) - 1
            log.append({
                "Entry": en.date(),
                "Exit": ex.date(),
                "Days": (ex-en).days,
                "Return": ret,
                "Type": "Win" if ret > 0 else "Loss"
            })
        except: pass
        
    return pd.DataFrame(log)

# --- UI ---
st.sidebar.header("1. Input")
f = st.sidebar.file_uploader("Upload", type=['xlsx','csv'])
df_raw = load_data(f)
if df_raw is None: st.info(f"Load Data... {DEFAULT_PATH}"); st.stop()
cols = df_raw.columns.tolist()

st.sidebar.header("2. Asset Config")
st.sidebar.markdown("**Risky Basket**")
assets = st.sidebar.multiselect("Select Assets", cols, default=cols[:2] if len(cols)>1 else cols)
weights = {}
if assets:
    def_w = 100/len(assets)
    for a in assets:
        weights[a] = st.sidebar.number_input(f"{a}%",0,100,int(def_w))/100.0
else: st.stop()

st.sidebar.markdown("**Safe Asset**")
d_guess = [c for c in cols if "Money" in c or "Liquid" in c or "Debt" in c]
safe_asset = st.sidebar.selectbox("Select Fund", cols, index=cols.index(d_guess[0]) if d_guess else 0)

st.sidebar.markdown("**Benchmark**")
n_guess = [c for c in cols if "Nifty" in c and "50" in c]
bench_col = st.sidebar.selectbox("Comparison", cols, index=cols.index(n_guess[0]) if n_guess else 0)

# --- SETTINGS ---
st.sidebar.header("3. Strategy Settings")
ma_type = st.sidebar.selectbox("MA Type", ["EMA (Exponential)", "SMA (Simple)"], index=0)
ma_window = st.sidebar.number_input("Period (Days)", value=200)
buffer_pct = st.sidebar.slider("Hysteresis Buffer (%)", 0.0, 5.0, 0.5, 0.1)

st.sidebar.header("4. Dates")
valid = df_raw.index
s_d = st.sidebar.date_input("Start", max(valid.min().date(), date(2014,1,1)))
e_d = st.sidebar.date_input("End", valid.max().date())

# --- RUN ---
mask = (df_raw.index.date >= s_d) & (df_raw.index.date <= e_d)
df_slice = df_raw.loc[mask]

strat, r_viz, ma_raw, sigs, df_used = run_strategy(
    df_raw, weights, safe_asset, buffer_pct, s_d, e_d, ma_type, ma_window
)

if strat is None: st.error("No Data"); st.stop()

# Stats
t_log = analyze_trades(sigs, strat)
bench = df_slice[bench_col]
bench = (bench/bench.iloc[0])*100
combined = pd.DataFrame({'Strategy': strat, 'Nifty 50': bench})

# --- METRICS ---
def metrics(s):
    if s.empty: return 0,0,0
    y = (s.index[-1]-s.index[0]).days/365.25
    cagr = (s.iloc[-1]/s.iloc[0])**(1/y)-1 if y>0 else 0
    vol = s.pct_change().dropna().std()*np.sqrt(252)
    dd = ((s/s.cummax())-1).min()
    return cagr, vol, dd

sc, sv, sd = metrics(combined['Strategy'])
bc, bv, bd = metrics(combined['Nifty 50'])

st.title(f"{ma_window} {ma_type.split(' ')[0]} Strategy")

col1, col2, col3, col4 = st.columns(4)
col1.metric("CAGR", f"{sc:.2%}", f"{(sc-bc)*100:.2f} pts")
col2.metric("Max Drawdown", f"{sd:.2%}", f"{(sd-bd)*100:.2f} pts", delta_color="inverse")
col3.metric("Volatility", f"{sv:.2%}", f"{(sv-bv)*100:.2f} pts", delta_color="inverse")
col4.metric("Trades", f"{len(t_log)}", f"Win Rate: {(len(t_log[t_log['Return']>0])/len(t_log) if len(t_log)>0 else 0):.0%}")

# --- CHARTS ---
st.subheader("Growth vs Nifty")
st.plotly_chart(px.line(combined, title="Portfolio Growth"), use_container_width=True)

with st.expander("Show Technical Chart (Entry/Exit Points)", expanded=True):
    # Scale MA to Viz
    scale_factor = r_viz.iloc[0] / ma_raw.iloc[0] # Approximate scaling
    # Recalculate MA on the rebased viz data for perfect alignment
    if "EMA" in ma_type:
        ma_viz = r_viz.ewm(span=ma_window, adjust=False).mean()
    else:
        ma_viz = r_viz.rolling(window=ma_window).mean()
        
    upper = ma_viz * (1 + buffer_pct/100)
    lower = ma_viz * (1 - buffer_pct/100)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r_viz.index, y=r_viz, name="Price", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=ma_viz.index, y=ma_viz, name=f"{ma_window} {ma_type.split(' ')[0]}", line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=upper.index, y=upper, name="Buy Threshold", line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=lower.index, y=lower, name="Sell Threshold", line=dict(color='red', dash='dot')))
    
    # Add Markers for Trades
    entries = t_log['Entry']
    exits = t_log['Exit']
    
    # Filter for visible range
    entries = [d for d in entries if d >= r_viz.index[0].date()]
    exits = [d for d in exits if d >= r_viz.index[0].date()]
    
    # To plot markers, we need Y values. Use interp or nearest.
    # Simple approach: Loop and find nearest Y
    # (Omitted for speed, lines are sufficient usually)

    st.plotly_chart(fig, use_container_width=True)

# --- TABLES ---
st.subheader("Yearly Returns")
y_df = combined.resample('YE').last()
s_row = pd.DataFrame([100,100], index=['Strategy','Nifty 50'], columns=[combined.index[0]-pd.Timedelta(days=1)]).T
y_calc = pd.concat([s_row, y_df]).sort_index().pct_change().dropna()
y_calc['Alpha'] = y_calc['Strategy'] - y_calc['Nifty 50']
y_calc.index = y_calc.index.year
st.dataframe(y_calc.style.format("{:.2%}").background_gradient(cmap='RdYlGn', subset=['Strategy','Alpha']), use_container_width=True)

st.subheader("Trade Log")
if not t_log.empty:
    st.dataframe(t_log.style.format({"Return": "{:.2%}"}).background_gradient(cmap='RdYlGn', subset=['Return']), use_container_width=True)
else:
    st.warning("Only 1 Trade detected (Buy & Hold). Try reducing the Buffer % or switching to EMA.")