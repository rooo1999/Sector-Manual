import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import date

st.set_page_config(page_title="MA Crossover Strategy", layout="wide")

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

def run_crossover_strategy(df_full, risky_weights, safe_asset, fast_period, slow_period, ma_type, start_date, end_date):
    
    # 1. Build Basket (Full History)
    risky_nav = get_basket_nav(df_full, risky_weights)
    
    # 2. Calculate MAs (Full History)
    if ma_type == "EMA":
        fast_ma = risky_nav.ewm(span=fast_period, adjust=False).mean()
        slow_ma = risky_nav.ewm(span=slow_period, adjust=False).mean()
    else:
        fast_ma = risky_nav.rolling(window=fast_period, min_periods=1).mean()
        slow_ma = risky_nav.rolling(window=slow_period, min_periods=1).mean()
    
    # 3. Logic: Invested if Fast > Slow
    # Shift(1) to avoid lookahead
    raw_signal = (fast_ma > slow_ma)
    trade_signal = raw_signal.shift(1).fillna(True)
    
    # 4. Slice Data
    mask = (df_full.index.date >= start_date) & (df_full.index.date <= end_date)
    df_slice = df_full.loc[mask]
    
    if df_slice.empty: return None, None, None, None, None, None

    risky_nav = risky_nav.loc[mask]
    trade_signal = trade_signal.loc[mask]
    fast_ma = fast_ma.loc[mask]
    slow_ma = slow_ma.loc[mask]
    
    # Rebase for Viz
    rebase = 100 / risky_nav.iloc[0]
    risky_viz = risky_nav * rebase
    fast_viz = fast_ma * rebase
    slow_viz = slow_ma * rebase
    
    # 5. Returns
    risky_ret = risky_nav.pct_change().fillna(0)
    safe_ret = df_slice[safe_asset].pct_change().fillna(0)
    
    final_ret = np.where(trade_signal, risky_ret, safe_ret)
    strat_nav = (1 + final_ret).cumprod() * 100
    strat_series = pd.Series(strat_nav, index=df_slice.index, name="Strategy")
    
    return strat_series, risky_viz, fast_viz, slow_viz, trade_signal, df_slice

def analyze_trades(signal_series, strategy_nav):
    # Convert to int
    sig_int = signal_series.astype(int)
    trades = sig_int.diff().fillna(0)
    
    entries = trades[trades == 1].index
    exits = trades[trades == -1].index
    
    # Handle Start/End
    if sig_int.iloc[0] == 1: entries = entries.insert(0, sig_int.index[0])
    if sig_int.iloc[-1] == 1: exits = exits.append(pd.Index([sig_int.index[-1]]))
        
    n = min(len(entries), len(exits))
    entries, exits = entries[:n], exits[:n]
    
    log = []
    total_days_held = 0
    
    for en, ex in zip(entries, exits):
        try:
            val_in = strategy_nav.loc[en]
            val_out = strategy_nav.loc[ex]
            ret = (val_out/val_in) - 1
            days = (ex - en).days
            total_days_held += days
            
            log.append({
                "Entry": en.date(),
                "Exit": ex.date(),
                "Days Held": days,
                "Return": ret,
                "Status": "Win" if ret > 0 else "Loss"
            })
        except: pass
        
    avg_hold = total_days_held / n if n > 0 else 0
    return pd.DataFrame(log), avg_hold

# --- 3. UI ---
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
st.sidebar.header("3. Crossover Settings")
ma_type = st.sidebar.selectbox("MA Type", ["EMA", "SMA"], index=0)
c1, c2 = st.sidebar.columns(2)
fast_p = c1.number_input("Fast MA (Green Line)", value=50, step=10, help="Entry Signal")
slow_p = c2.number_input("Slow MA (Red Line)", value=200, step=10, help="Trend Baseline")

st.sidebar.caption("Strategy: Buy when FAST > SLOW. Sell when FAST < SLOW.")

st.sidebar.header("4. Dates")
valid = df_raw.index
s_d = st.sidebar.date_input("Start", max(valid.min().date(), date(2014,1,1)))
e_d = st.sidebar.date_input("End", valid.max().date())

# --- RUN ---
mask = (df_raw.index.date >= s_d) & (df_raw.index.date <= e_d)
df_slice = df_raw.loc[mask]

strat, r_viz, fast_viz, slow_viz, sigs, df_used = run_crossover_strategy(
    df_raw, weights, safe_asset, int(fast_p), int(slow_p), ma_type, s_d, e_d
)

if strat is None: st.error("No Data"); st.stop()

# Stats
t_log, avg_hold_days = analyze_trades(sigs, strat)
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

st.title(f"{fast_p}/{slow_p} {ma_type} Crossover Strategy")

col1, col2, col3, col4 = st.columns(4)
col1.metric("CAGR", f"{sc:.2%}", f"{(sc-bc)*100:.2f} pts")
col2.metric("Max Drawdown", f"{sd:.2%}", f"{(sd-bd)*100:.2f} pts", delta_color="inverse")
col3.metric("Trades", f"{len(t_log)}", f"Avg Hold: {int(avg_hold_days)} Days")
col4.metric("Win Rate", f"{(len(t_log[t_log['Return']>0])/len(t_log) if len(t_log)>0 else 0):.0%}")

# --- CHARTS ---
st.subheader("Performance vs Benchmark")
st.plotly_chart(px.line(combined, title="Growth of 100"), use_container_width=True)

with st.expander("🔍 Crossover Visualization (The Golden Cross)", expanded=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r_viz.index, y=r_viz, name="Price", line=dict(color='gray', width=1)))
    fig.add_trace(go.Scatter(x=fast_viz.index, y=fast_viz, name=f"Fast MA ({fast_p})", line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=slow_viz.index, y=slow_viz, name=f"Slow MA ({slow_p})", line=dict(color='red', width=2)))
    
    # Shade zones
    y_min = r_viz.min()
    y_max = r_viz.max()
    
    # We can visualize 'Invested' zones by coloring background or using marker lines
    # Simple method: Dots on crossovers
    crossovers = sigs.astype(int).diff()
    buys = crossovers[crossovers == 1].index
    sells = crossovers[crossovers == -1].index
    
    # Filter for view
    buys = [b for b in buys if b >= s_d]
    sells = [s for s in sells if s >= s_d]

    # Add markers
    # For accurate Y value, grab Fast MA value at that date
    if buys:
        buy_y = [fast_viz.loc[b] for b in buys]
        fig.add_trace(go.Scatter(x=buys, y=buy_y, mode='markers', name="Golden Cross (Buy)", marker=dict(color='green', size=12, symbol='triangle-up')))
    if sells:
        sell_y = [fast_viz.loc[s] for s in sells]
        fig.add_trace(go.Scatter(x=sells, y=sell_y, mode='markers', name="Death Cross (Sell)", marker=dict(color='red', size=12, symbol='triangle-down')))

    st.plotly_chart(fig, use_container_width=True)

# --- TABLES ---
st.subheader("Yearly Returns")
y_df = combined.resample('YE').last()
s_row = pd.DataFrame([100,100], index=['Strategy','Nifty 50'], columns=[combined.index[0]-pd.Timedelta(days=1)]).T
y_calc = pd.concat([s_row, y_df]).sort_index().pct_change().dropna()
y_calc['Alpha'] = y_calc['Strategy'] - y_calc['Nifty 50']
y_calc.index = y_calc.index.year
st.dataframe(y_calc.style.format("{:.2%}").background_gradient(cmap='RdYlGn', subset=['Strategy','Alpha']), use_container_width=True)

st.subheader("Trade Log (Proof of Holding Period)")
if not t_log.empty:
    st.dataframe(t_log.style.format({"Return": "{:.2%}"}).background_gradient(cmap='RdYlGn', subset=['Return']), use_container_width=True)