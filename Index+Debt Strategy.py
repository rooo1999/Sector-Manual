import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import date

st.set_page_config(page_title="200 DMA Optimizer", layout="wide")

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
    
    # Monthly Rebalancing of the Basket itself
    for i in range(1, len(idx)):
        vals = vals * (1 + ret_arr[i])
        total = np.sum(vals)
        if idx[i].month != idx[i-1].month: 
            vals = total * target_w
        hist.append(total)
        dates.append(idx[i])
        
    return pd.Series(hist, index=dates, name="Risky_Basket")

def run_buffered_200dma(df_full, risky_weights, safe_asset, buffer_pct, start_date, end_date):
    """
    Calculates 200 DMA on full history, applies Buffer logic, then slices results.
    """
    # 1. Build Basket (Full History)
    risky_nav = get_basket_nav(df_full, risky_weights)
    
    # 2. Calculate 200 SMA (Full History)
    # min_periods=1 ensures we get values even if history is short
    ma_200 = risky_nav.rolling(window=200, min_periods=1).mean()
    
    # 3. Apply Buffer Logic (State Machine)
    # State 1 = Invested, 0 = Cash
    price = risky_nav.values
    ma = ma_200.values
    
    signals = []
    state = 1 # Start Invested (Optimistic default)
    
    buffer_mult = buffer_pct / 100.0
    
    for i in range(len(price)):
        current_p = price[i]
        current_ma = ma[i]
        
        if np.isnan(current_ma):
            signals.append(True) # Stay invested if no MA yet
            continue
            
        upper_band = current_ma * (1 + buffer_mult)
        lower_band = current_ma * (1 - buffer_mult)
        
        if state == 1:
            # We are Invested. Look for Exit.
            # Exit only if Price < MA - Buffer
            if current_p < lower_band:
                state = 0
        else:
            # We are in Cash. Look for Entry.
            # Enter only if Price > MA + Buffer
            if current_p > upper_band:
                state = 1
                
        signals.append(bool(state))
        
    raw_signal_series = pd.Series(signals, index=risky_nav.index)
    
    # 4. Shift Signal (Trade execution happens next day)
    trade_signal = raw_signal_series.shift(1).fillna(True)
    
    # 5. Slice Data to User Range
    mask = (df_full.index.date >= start_date) & (df_full.index.date <= end_date)
    df_slice = df_full.loc[mask]
    
    if df_slice.empty: return None, None, None, None

    risky_nav = risky_nav.loc[mask]
    trade_signal = trade_signal.loc[mask]
    ma_200 = ma_200.loc[mask]
    
    # Rebase visual NAV
    risky_nav_viz = (risky_nav / risky_nav.iloc[0]) * 100
    
    # 6. Calculate Strategy Returns
    risky_ret = risky_nav.pct_change().fillna(0)
    safe_ret = df_slice[safe_asset].pct_change().fillna(0)
    
    final_ret = np.where(trade_signal, risky_ret, safe_ret)
    strat_nav = (1 + final_ret).cumprod() * 100
    strat_series = pd.Series(strat_nav, index=df_slice.index, name="Strategy")
    
    return strat_series, risky_nav_viz, ma_200, trade_signal

def analyze_trades(signal_series, strategy_nav):
    """
    Reconstructs trade history from signals.
    """
    trades = signal_series.diff().fillna(0)
    entries = trades[trades == 1].index
    exits = trades[trades == -1].index
    
    # Handle open positions
    if signal_series.iloc[0] == True:
        entries = entries.insert(0, signal_series.index[0])
    if signal_series.iloc[-1] == True:
        exits = exits.append(pd.Index([signal_series.index[-1]]))
        
    # Align
    n = min(len(entries), len(exits))
    entries = entries[:n]
    exits = exits[:n]
    
    trade_log = []
    for en, ex in zip(entries, exits):
        # NAV at entry vs NAV at exit
        try:
            val_in = strategy_nav.loc[en]
            val_out = strategy_nav.loc[ex]
            ret = (val_out / val_in) - 1
            days = (ex - en).days
            trade_log.append({
                "Entry Date": en.date(),
                "Exit Date": ex.date(),
                "Days Held": days,
                "Return": ret,
                "Result": "Win" if ret > 0 else "Loss"
            })
        except: pass
        
    return pd.DataFrame(trade_log)

# --- UI ---
st.sidebar.header("1. Input Data")
f = st.sidebar.file_uploader("Upload", type=['xlsx','csv'])
df_raw = load_data(f)
if df_raw is None: st.info(f"Load Data... {DEFAULT_PATH}"); st.stop()
cols = df_raw.columns.tolist()

st.sidebar.header("2. Asset Config")
st.sidebar.markdown("**Risky Basket (Equity)**")
assets = st.sidebar.multiselect("Select Assets", cols, default=cols[:2] if len(cols)>1 else cols)
weights = {}
if assets:
    def_w = 100/len(assets)
    for a in assets:
        weights[a] = st.sidebar.number_input(f"{a}%",0,100,int(def_w))/100.0
else: st.stop()

st.sidebar.markdown("**Safe Asset (Debt)**")
d_guess = [c for c in cols if "Money" in c or "Liquid" in c or "Debt" in c]
safe_asset = st.sidebar.selectbox("Select Fund", cols, index=cols.index(d_guess[0]) if d_guess else 0)

st.sidebar.markdown("**Benchmark (Nifty)**")
n_guess = [c for c in cols if "Nifty" in c and "50" in c]
bench_col = st.sidebar.selectbox("Comparison", cols, index=cols.index(n_guess[0]) if n_guess else 0)

# 3. Strategy Settings
st.sidebar.header("3. 200 DMA Settings")
buffer_pct = st.sidebar.slider("Whipsaw Buffer (%)", 0.0, 5.0, 0.5, 0.1, help="Only trade if Price crosses MA by this %")

st.sidebar.header("4. Dates")
valid = df_raw.index
s_d = st.sidebar.date_input("Start", max(valid.min().date(), date(2014,1,1)))
e_d = st.sidebar.date_input("End", valid.max().date())

# --- RUN ---
mask = (df_raw.index.date >= s_d) & (df_raw.index.date <= e_d)
df_slice = df_raw.loc[mask]

strat_series, risky_viz, ma_viz, signals = run_buffered_200dma(
    df_raw, weights, safe_asset, buffer_pct, s_d, e_d
)

if strat_series is None: st.error("No Data"); st.stop()

bench_series = df_slice[bench_col]
bench_series = (bench_series / bench_series.iloc[0]) * 100

combined = pd.DataFrame({'Strategy': strat_series, 'Nifty 50': bench_series})

# Trade Stats
trade_df = analyze_trades(signals, strat_series)

# --- DASHBOARD ---
st.title("Optimized 200 DMA Strategy")

# Metrics
def get_metrics(s):
    if s.empty: return 0,0,0
    y = (s.index[-1]-s.index[0]).days/365.25
    cagr = (s.iloc[-1]/s.iloc[0])**(1/y)-1 if y>0 else 0
    vol = s.pct_change().dropna().std()*np.sqrt(252)
    dd = ((s/s.cummax())-1).min()
    return cagr, vol, dd

sc, sv, sd = get_metrics(combined['Strategy'])
bc, bv, bd = get_metrics(combined['Nifty 50'])

# Top Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Strategy CAGR", f"{sc:.2%}", f"{(sc-bc)*100:.2f} pts")
col2.metric("Max Drawdown", f"{sd:.2%}", f"{(sd-bd)*100:.2f} pts", delta_color="inverse")
col3.metric("Volatility", f"{sv:.2%}", f"{(sv-bv)*100:.2f} pts", delta_color="inverse")
col4.metric("Total Trades", f"{len(trade_df)}", f"Win Rate: {(len(trade_df[trade_df['Return']>0])/len(trade_df) if len(trade_df)>0 else 0):.0%}")

# Chart 1: Performance
st.subheader("Performance vs Nifty 50")
st.plotly_chart(px.line(combined, title="Growth of 100"), use_container_width=True)

# Chart 2: Technicals (Price vs MA)
with st.expander("🔍 View Technical Chart (Price vs 200 DMA)"):
    fig_tech = go.Figure()
    fig_tech.add_trace(go.Scatter(x=risky_viz.index, y=risky_viz, name="Basket Price", line=dict(color='blue')))
    # Scale MA to match rebased price
    scale = risky_viz.iloc[0] / ma_viz.iloc[0] # Approx scaling for viz
    # Better: Calculate MA on the rebased viz series directly for accurate plotting
    ma_viz_rebased = risky_viz.rolling(200, min_periods=1).mean()
    
    # Add Bands
    upper = ma_viz_rebased * (1 + buffer_pct/100)
    lower = ma_viz_rebased * (1 - buffer_pct/100)
    
    fig_tech.add_trace(go.Scatter(x=risky_viz.index, y=ma_viz_rebased, name="200 DMA", line=dict(color='orange')))
    fig_tech.add_trace(go.Scatter(x=risky_viz.index, y=upper, name="Buy Line", line=dict(color='green', dash='dot', width=1)))
    fig_tech.add_trace(go.Scatter(x=risky_viz.index, y=lower, name="Sell Line", line=dict(color='red', dash='dot', width=1)))
    
    # Shade cash zones
    cash_zones = signals[signals==False]
    if not cash_zones.empty:
         fig_tech.add_trace(go.Scatter(x=cash_zones.index, y=[risky_viz.min()]*len(cash_zones), 
                                          mode='markers', name="In Cash", marker=dict(color='red', symbol='square')))
    
    st.plotly_chart(fig_tech, use_container_width=True)

# Yearly Table
st.subheader("Yearly Returns Comparison")
yearly = combined.resample('YE').last()
start_row = pd.DataFrame([100, 100], index=['Strategy','Nifty 50'], columns=[combined.index[0]-pd.Timedelta(days=1)]).T
y_calc = pd.concat([start_row, yearly]).sort_index()
y_ret = y_calc.pct_change().dropna()
y_ret['Outperformance'] = y_ret['Strategy'] - y_ret['Nifty 50']
y_ret.index = y_ret.index.year

st.dataframe(y_ret.style.format("{:.2%}").background_gradient(cmap='RdYlGn', subset=['Strategy', 'Outperformance']), use_container_width=True)

# Trade Log
st.subheader("Trade History Log")
if not trade_df.empty:
    st.dataframe(trade_df.style.format({"Return": "{:.2%}"}).background_gradient(cmap='RdYlGn', subset=['Return']), use_container_width=True)
else:
    st.info("No completed trades in this period (Strategy remained in one state).")