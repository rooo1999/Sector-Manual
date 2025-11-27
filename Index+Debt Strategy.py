import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import date

st.set_page_config(page_title="Regime Filter Pro", layout="wide")

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

# --- ADVANCED SIGNAL LOGIC ---

def calculate_buffered_signal(price_series, ma_series, entry_buffer_pct, exit_buffer_pct):
    """
    Generates signals with Hysteresis (Separate Entry/Exit).
    Entry: Price > MA * (1 + entry_buffer)
    Exit:  Price < MA * (1 - exit_buffer)
    """
    signals = []
    
    # Initial State: We assume Risk On (True) for the start (First 200 days)
    # 1 = Invested (Risk On), 0 = Safe (Risk Off)
    current_state = 1 
    
    prices = price_series.values
    mas = ma_series.values
    
    for i in range(len(prices)):
        p = prices[i]
        m = mas[i]
        
        # Handle NaN MAs (First 200 days) -> Force Invested
        if np.isnan(m):
            signals.append(True)
            continue
            
        # Define Bands
        upper_band = m * (1 + entry_buffer_pct/100.0)
        lower_band = m * (1 - exit_buffer_pct/100.0)
        
        if current_state == 1: # Currently Invested
            if p < lower_band:
                current_state = 0 # Sell Signal
        else: # Currently in Cash
            if p > upper_band:
                current_state = 1 # Re-entry Signal
                
        signals.append(bool(current_state))
        
    return pd.Series(signals, index=price_series.index)

def construct_basket(df_full, weights_dict, rebalance_freq):
    assets = list(weights_dict.keys())
    returns = df_full[assets].pct_change().fillna(0)
    
    vals = np.array([100.0 * weights_dict[a] for a in assets])
    hist = [100.0]
    dates = [returns.index[0]]
    ret_arr = returns.values
    target_w = np.array([weights_dict[a] for a in assets])
    idx = returns.index
    
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

def run_strategy_engine(df_full, mode, risky_weights, safe_asset, ma_days, rebal_freq, 
                       start_date, end_date, signal_source_col, entry_buf, exit_buf):
    
    # 1. Risky Basket
    risky_nav = construct_basket(df_full, risky_weights, rebal_freq)
    
    # 2. Signal Generation (Full History)
    if mode == "Trend Following" and signal_source_col:
        signal_price = df_full[signal_source_col]
    else:
        signal_price = risky_nav

    # Calculate MA
    ma_series = signal_price.rolling(window=ma_days, min_periods=1).mean()
    
    # Calculate Buffered Signals (Loop based)
    # We shift(1) INSIDE the logic? No, we calculate state based on today's close, 
    # then shift the RESULT by 1 to trade tomorrow.
    raw_signal_series = calculate_buffered_signal(signal_price, ma_series, entry_buf, exit_buf)
    
    # Shift by 1 day to avoid lookahead (Trade on Next Open based on Today Close)
    trade_signal = raw_signal_series.shift(1).fillna(True)
    
    # 3. Slice
    mask = (df_full.index.date >= start_date) & (df_full.index.date <= end_date)
    df_slice = df_full.loc[mask]
    
    if df_slice.empty: return None, None, None, None, None

    risky_nav = risky_nav.loc[mask]
    trade_signal = trade_signal.loc[mask]
    signal_price = signal_price.loc[mask]
    ma_series = ma_series.loc[mask]
    
    # Visual Rebase
    risky_nav_viz = (risky_nav / risky_nav.iloc[0]) * 100
    
    # 4. Returns
    risky_ret = risky_nav.pct_change().fillna(0)
    safe_ret = df_slice[safe_asset].pct_change().fillna(0)
    
    if mode == "Fixed Allocation":
        final_ret = risky_ret 
    else:
        final_ret = np.where(trade_signal == True, risky_ret, safe_ret)
        
    strat_nav = (1 + final_ret).cumprod() * 100
    strat_series = pd.Series(strat_nav, index=df_slice.index, name="Strategy")
    
    return strat_series, df_slice, trade_signal, signal_price, ma_series

# --- UI ---
st.sidebar.header("1. Data")
f = st.sidebar.file_uploader("Upload", type=['xlsx','csv'])
df_raw = load_data(f)
if df_raw is None: st.info(f"Load Data... {DEFAULT_PATH}"); st.stop()
cols = df_raw.columns.tolist()

# --- SETTINGS ---
st.sidebar.header("2. Logic")
mode = st.sidebar.selectbox("Mode", ["Trend Following", "Fixed Allocation"])

def get_weights(label):
    st.sidebar.markdown(f"**{label}**")
    assets = st.sidebar.multiselect("Assets", cols, key=label)
    w_dict = {}
    if assets:
        def_w = 100/len(assets)
        tot=0
        for a in assets:
            v = st.sidebar.number_input(f"{a}%",0,100,int(def_w),key=f"w{a}{label}")
            w_dict[a]=v/100.0
            tot+=v
        if tot!=100: st.sidebar.error("Sum != 100%")
    return w_dict

risky_weights = {}
safe_asset = cols[0]
signal_source = None
entry_buf = 0.0
exit_buf = 0.0
ma_days = 200

if mode == "Fixed Allocation":
    risky_weights = get_weights("Portfolio")
    rebal_freq = st.sidebar.selectbox("Rebal", ["Daily","Monthly","Yearly"])
else:
    risky_weights = get_weights("Risky Basket (Equity)")
    rebal_freq = st.sidebar.selectbox("Basket Rebal", ["Daily","Monthly","Yearly"], index=1)
    
    st.sidebar.markdown("---")
    d_guess = [c for c in cols if "Money" in c or "Liquid" in c or "Debt" in c]
    safe_asset = st.sidebar.selectbox("Safe Asset", cols, index=cols.index(d_guess[0]) if d_guess else 0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Traffic Light Logic**")
    sig_type = st.sidebar.radio("Signal Source", ["Broad Market (Nifty)", "Self (Basket)"])
    if sig_type == "Broad Market (Nifty)":
        n_guess = [c for c in cols if "Nifty" in c and "50" in c]
        signal_source = st.sidebar.selectbox("Index", cols, index=cols.index(n_guess[0]) if n_guess else 0)
    
    c1, c2 = st.sidebar.columns(2)
    ma_days = c1.number_input("DMA", 200)
    
    st.sidebar.markdown("**Buffers (Avoid Whipsaws)**")
    c3, c4 = st.sidebar.columns(2)
    entry_buf = c3.number_input("Entry Buffer (%)", 0.0, 10.0, 0.0, 0.5, help="Re-enter only if Price > DMA + X%")
    exit_buf = c4.number_input("Exit Buffer (%)", 0.0, 10.0, 0.0, 0.5, help="Exit only if Price < DMA - X%")
    
    if entry_buf > 0 or exit_buf > 0:
        st.sidebar.caption(f"Strategy: Buy > {ma_days}DMA + {entry_buf}%. Sell < {ma_days}DMA - {exit_buf}%.")

# Date
st.sidebar.markdown("---")
bench_col = st.sidebar.selectbox("Benchmark", cols, index=0)
valid = df_raw.index
s_d = st.sidebar.date_input("Start", max(valid.min().date(), date(2014,1,1)))
e_d = st.sidebar.date_input("End", valid.max().date())

# --- RUN ---
if not risky_weights or sum(risky_weights.values()) != 1.0: st.stop()

strat, df_s, sig_vec, sig_price, sig_ma = run_strategy_engine(
    df_raw, mode, risky_weights, safe_asset, ma_days, rebal_freq, s_d, e_d, signal_source, entry_buf, exit_buf
)

if strat is None: st.error("No Data"); st.stop()

bench = df_s[bench_col]
bench = (bench/bench.iloc[0])*100
final = pd.DataFrame({'Strategy': strat, 'Benchmark': bench})

# --- METRICS ---
st.title("Pro Strategy Dashboard")

def stats(s):
    if s.empty: return 0,0,0
    y = (s.index[-1]-s.index[0]).days/365.25
    if y<=0: return 0,0,0
    cagr = (s.iloc[-1]/s.iloc[0])**(1/y)-1
    vol = s.pct_change().dropna().std()*np.sqrt(252)
    dd = ((s/s.cummax())-1).min()
    return cagr, vol, dd

sc, sv, sd = stats(final['Strategy'])
bc, bv, bd = stats(final['Benchmark'])

k1, k2, k3 = st.columns(3)
k1.metric("CAGR", f"{sc:.2%}", f"{(sc-bc)*100:.2f}")
k2.metric("Max DD", f"{sd:.2%}", f"{(sd-bd)*100:.2f}", delta_color="inverse")
k3.metric("Vol", f"{sv:.2%}", f"{(sv-bv)*100:.2f}", delta_color="inverse")

st.subheader("Performance")
st.plotly_chart(px.line(final, title="Growth of 100"), use_container_width=True)

# --- TRAFFIC LIGHT CHART ---
if mode == "Trend Following":
    st.subheader("🚦 Traffic Light Analysis")
    with st.expander("Show Signal Chart", expanded=True):
        # We construct a DF with Raw Levels
        sig_viz = pd.DataFrame({
            'Market Price': sig_price,
            f'{ma_days} DMA': sig_ma
        })
        
        # Add Buffer Lines for Visual Reference if buffers exist
        if entry_buf > 0:
            sig_viz['Buy Threshold'] = sig_ma * (1 + entry_buf/100)
        if exit_buf > 0:
            sig_viz['Sell Threshold'] = sig_ma * (1 - exit_buf/100)
            
        # Plot using Graph Objects for better control (Raw Levels)
        fig_sig = go.Figure()
        
        # 1. Market Price
        fig_sig.add_trace(go.Scatter(x=sig_viz.index, y=sig_viz['Market Price'], name=f"Price ({signal_source if signal_source else 'Basket'})", line=dict(color='blue')))
        
        # 2. DMA
        fig_sig.add_trace(go.Scatter(x=sig_viz.index, y=sig_viz[f'{ma_days} DMA'], name=f"{ma_days} DMA", line=dict(color='orange', width=2)))
        
        # 3. Buffers
        if entry_buf > 0:
             fig_sig.add_trace(go.Scatter(x=sig_viz.index, y=sig_viz['Buy Threshold'], name="Buy Line", line=dict(color='green', dash='dot')))
        if exit_buf > 0:
             fig_sig.add_trace(go.Scatter(x=sig_viz.index, y=sig_viz['Sell Threshold'], name="Sell Line", line=dict(color='red', dash='dot')))

        # 4. Background Color for Regime
        # We can simulate this by shading areas, but for performance, let's keep it line based
        fig_sig.update_layout(title="Signal Source: Raw Levels & Moving Averages", hovermode="x unified")
        st.plotly_chart(fig_sig, use_container_width=True)
        
        # State Table
        st.write(f"**Current State:** {'🟢 Equity (Risk On)' if sig_vec.iloc[-1] else '🔴 Cash (Risk Off)'}")

# --- TABLES ---
t1, t2 = st.tabs(["Heatmap", "Yearly"])
with t1:
    m = final['Strategy'].resample('ME').apply(lambda x: x.iloc[-1]/x.iloc[0]-1 if len(x)>0 else 0)
    p = pd.DataFrame({'Y': m.index.year, 'M': m.index.strftime('%b'), 'V': m.values}).pivot(index='Y', columns='M', values='V')
    p = p.reindex(columns=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    p['YTD'] = final['Strategy'].resample('YE').apply(lambda x: x.iloc[-1]/x.iloc[0]-1).values
    st.dataframe(p.style.format("{:.2%}", na_rep="-").background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)

with t2:
    y = final.resample('YE').last()
    s_r = pd.DataFrame([100, 100], index=['Strategy','Benchmark'], columns=[final.index[0]-pd.Timedelta(days=1)]).T
    c = pd.concat([s_r, y]).sort_index()
    yr = c.pct_change().dropna()
    yr['Alpha'] = yr['Strategy'] - yr['Benchmark']
    yr.index = yr.index.year
    st.dataframe(yr.style.format("{:.2%}").background_gradient(cmap='RdYlGn', subset=['Strategy','Alpha']), use_container_width=True)