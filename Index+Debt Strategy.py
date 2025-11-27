import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import date

st.set_page_config(page_title="Strategy Debugger", layout="wide")

# --- 1. DATA LOADER ---
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

# --- 2. CALCULATIONS ---

def get_basket_nav(df, weights):
    assets = list(weights.keys())
    # Ensure we only use rows where we have data
    valid_df = df[assets].dropna()
    returns = valid_df.pct_change().fillna(0)
    
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
    # FORCE INT to prevent crashes
    window = int(window)
    if ma_type == "EMA (Exponential)":
        return series.ewm(span=window, adjust=False).mean()
    else:
        return series.rolling(window=window, min_periods=1).mean()

def run_strategy(df_full, risky_weights, safe_asset, buffer_pct, start_date, end_date, ma_type, ma_window):
    
    # 1. Full History Basket
    risky_nav = get_basket_nav(df_full, risky_weights)
    
    # 2. Full History MA
    ma_series = calculate_ma(risky_nav, ma_window, ma_type)
    
    # 3. Logic Loop
    price = risky_nav.values
    ma = ma_series.values
    
    signals = []
    state = 1 
    
    buffer_mult = buffer_pct / 100.0
    
    for i in range(len(price)):
        p = price[i]
        m = ma[i]
        
        # Handle start logic safely
        if np.isnan(m):
            signals.append(True)
            continue
            
        upper = m * (1 + buffer_mult)
        lower = m * (1 - buffer_mult)
        
        if state == 1:
            if p < lower:
                state = 0
        else:
            if p > upper:
                state = 1
                
        signals.append(bool(state))
        
    raw_signal = pd.Series(signals, index=risky_nav.index)
    
    # 4. Shift Signal
    trade_signal = raw_signal.shift(1).fillna(True)
    
    # 5. Slicing
    mask = (df_full.index.date >= start_date) & (df_full.index.date <= end_date)
    df_slice = df_full.loc[mask]
    
    if df_slice.empty: return None, None, None, None, None

    # Slice Everything
    risky_nav_sliced = risky_nav.loc[mask]
    trade_signal_sliced = trade_signal.loc[mask]
    ma_series_sliced = ma_series.loc[mask]
    
    # --- VISUALIZATION REBASING ---
    # We rebase Price to 100. We must rebase MA by the exact same factor to keep them aligned.
    rebase_factor = 100 / risky_nav_sliced.iloc[0]
    
    risky_viz = risky_nav_sliced * rebase_factor
    ma_viz = ma_series_sliced * rebase_factor
    
    # 6. Returns
    risky_ret = risky_nav_sliced.pct_change().fillna(0)
    safe_ret = df_slice[safe_asset].pct_change().fillna(0)
    
    final_ret = np.where(trade_signal_sliced, risky_ret, safe_ret)
    strat_nav = (1 + final_ret).cumprod() * 100
    strat_series = pd.Series(strat_nav, index=df_slice.index, name="Strategy")
    
    return strat_series, risky_viz, ma_viz, trade_signal_sliced, df_slice

def analyze_trades(signal_series, strategy_nav):
    # Convert to INT to ensure diff works correctly (True-False can vary by version)
    sig_int = signal_series.astype(int)
    trades = sig_int.diff().fillna(0)
    
    entries = trades[trades == 1].index
    exits = trades[trades == -1].index
    
    # Handle Start
    if sig_int.iloc[0] == 1:
        entries = entries.insert(0, sig_int.index[0])
        
    # Handle End
    if sig_int.iloc[-1] == 1:
        exits = exits.append(pd.Index([sig_int.index[-1]]))
        
    # Align
    n = min(len(entries), len(exits))
    entries = entries[:n]
    exits = exits[:n]
    
    log = []
    for en, ex in zip(entries, exits):
        try:
            val_in = strategy_nav.loc[en]
            val_out = strategy_nav.loc[ex]
            ret = (val_out/val_in) - 1
            days = (ex - en).days
            log.append({
                "Entry": en.date(),
                "Exit": ex.date(),
                "Days": days,
                "Return": ret,
                "Status": "Win" if ret > 0 else "Loss"
            })
        except: pass
        
    return pd.DataFrame(log)

# --- 3. UI ---
st.sidebar.header("1. Input")
f = st.sidebar.file_uploader("Upload", type=['xlsx','csv'])
df_raw = load_data(f)
if df_raw is None: st.info(f"Load Data... {DEFAULT_PATH}"); st.stop()
cols = df_raw.columns.tolist()

st.sidebar.header("2. Composition")
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
ma_window = st.sidebar.number_input("Period (Days)", value=50, step=10) # Default 50 per request
buffer_pct = st.sidebar.slider("Hysteresis Buffer (%)", 0.0, 5.0, 0.0, 0.1) # Default 0.0 for sensitivity

st.sidebar.header("4. Dates")
valid = df_raw.index
s_d = st.sidebar.date_input("Start", max(valid.min().date(), date(2014,1,1)))
e_d = st.sidebar.date_input("End", valid.max().date())

# --- RUN ---
mask = (df_raw.index.date >= s_d) & (df_raw.index.date <= e_d)
df_slice = df_raw.loc[mask]

strat, r_viz, ma_viz, sigs, df_used = run_strategy(
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

st.title(f"{int(ma_window)} {ma_type.split(' ')[0]} Strategy Analysis")

col1, col2, col3, col4 = st.columns(4)
col1.metric("CAGR", f"{sc:.2%}", f"{(sc-bc)*100:.2f} pts")
col2.metric("Max Drawdown", f"{sd:.2%}", f"{(sd-bd)*100:.2f} pts", delta_color="inverse")
col3.metric("Win Rate", f"{(len(t_log[t_log['Return']>0])/len(t_log) if len(t_log)>0 else 0):.0%}", f"{len(t_log)} Trades")
col4.metric("Avg Trade Return", f"{(t_log['Return'].mean() if not t_log.empty else 0):.2%}")

# --- CHARTS ---
st.subheader("Performance vs Benchmark")
st.plotly_chart(px.line(combined, title="Growth of 100"), use_container_width=True)

with st.expander("🔍 Signal Logic Chart (Why did we buy/sell?)", expanded=True):
    # Calculate Bands
    upper = ma_viz * (1 + buffer_pct/100)
    lower = ma_viz * (1 - buffer_pct/100)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r_viz.index, y=r_viz, name="Basket Price", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=ma_viz.index, y=ma_viz, name=f"MA {int(ma_window)}", line=dict(color='orange')))
    
    if buffer_pct > 0:
        fig.add_trace(go.Scatter(x=upper.index, y=upper, name="Buy Zone", line=dict(color='green', dash='dot')))
        fig.add_trace(go.Scatter(x=lower.index, y=lower, name="Sell Zone", line=dict(color='red', dash='dot')))
    
    # Overlay Cash Zones
    cash_mask = sigs == False
    if cash_mask.any():
        cash_df = r_viz[cash_mask]
        fig.add_trace(go.Scatter(x=cash_df.index, y=[r_viz.min()]*len(cash_df), 
                                 mode='markers', name="In Cash", marker=dict(color='red', symbol='square')))
        
    st.plotly_chart(fig, use_container_width=True)

# --- DEBUG & LOGS ---
tab1, tab2 = st.tabs(["Trade Log", "Yearly Returns"])

with tab1:
    if not t_log.empty:
        st.dataframe(t_log.style.format({"Return": "{:.2%}"}).background_gradient(cmap='RdYlGn', subset=['Return']), use_container_width=True)
    else:
        st.warning("1 Trade Detected (Buy & Hold). This usually means the trend was strong or buffer was too wide.")

with tab2:
    y_df = combined.resample('YE').last()
    s_row = pd.DataFrame([100,100], index=['Strategy','Nifty 50'], columns=[combined.index[0]-pd.Timedelta(days=1)]).T
    y_calc = pd.concat([s_row, y_df]).sort_index().pct_change().dropna()
    y_calc['Alpha'] = y_calc['Strategy'] - y_calc['Nifty 50']
    y_calc.index = y_calc.index.year
    st.dataframe(y_calc.style.format("{:.2%}").background_gradient(cmap='RdYlGn', subset=['Strategy','Alpha']), use_container_width=True)