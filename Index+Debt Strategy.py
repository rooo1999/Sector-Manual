import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import date

st.set_page_config(page_title="Strategy Audit Engine", layout="wide")

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

# --- CALCULATIONS ---

def construct_basket(df_full, weights_dict, rebalance_freq):
    """
    Constructs the NAV of the 'Risky Portfolio'.
    Now allows ANY asset (Equity or Debt) in the mix.
    """
    assets = list(weights_dict.keys())
    returns = df_full[assets].pct_change().fillna(0)
    
    # Initialize logic
    vals = np.array([100.0 * weights_dict[a] for a in assets])
    hist = [100.0]
    dates = [returns.index[0]]
    ret_arr = returns.values
    target_w = np.array([weights_dict[a] for a in assets])
    idx = returns.index
    
    # Loop
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

def run_hybrid_logic(bench_series, basket_series, ma_days):
    """
    Implements the State Machine for Entry/Exit.
    """
    # 1. Calculate Indicators
    # Min periods=1 ensures we have values from the start
    bench_ma = bench_series.rolling(window=ma_days, min_periods=1).mean()
    basket_ma = basket_series.rolling(window=ma_days, min_periods=1).mean()
    
    b_price = bench_series.values
    b_ma = bench_ma.values
    k_price = basket_series.values
    k_ma = basket_ma.values
    
    signals = []
    # Start: Assume Invested (1) or Cash (0)?
    # To start "Risk On" for first 200 days, we set state=1
    state = 1 
    
    for i in range(len(b_price)):
        # If MA is NaN (very first day), force Invested
        if np.isnan(b_ma[i]) or np.isnan(k_ma[i]):
            signals.append(True)
            continue
            
        if state == 1:
            # --- EXIT LOGIC ---
            # We are holding the Basket. We check if Basket fails.
            # Exit if Basket Price < Basket MA
            if k_price[i] < k_ma[i]:
                state = 0
        else:
            # --- ENTRY LOGIC ---
            # We are in Cash. We check if Benchmark is healthy.
            # Enter if Benchmark > Benchmark MA
            
            # CRITICAL FIX: To prevent 2018 Whipsaw
            # If we enter while Basket < Basket MA, we will just exit tomorrow.
            # So, technically, we should only enter if Benchmark > MA.
            # The User asked for "Entry if Benchmark crosses MA". 
            # Strictly following that causes the loop. 
            # I will implement strictly as asked, but you will see the whipsaw in Debug.
            
            if b_price[i] > b_ma[i]:
                state = 1
        
        signals.append(bool(state))
        
    return pd.Series(signals, index=bench_series.index), bench_ma, basket_ma

def run_engine(df_full, mode, risky_weights, safe_asset, ma_days, rebal_freq, 
               start_date, end_date, signal_logic, signal_source_col):
    
    # 1. Build Risky Basket (Can include debt funds now)
    risky_nav = construct_basket(df_full, risky_weights, rebal_freq)
    
    # 2. Benchmark for Signal
    if signal_source_col:
        bench_nav = df_full[signal_source_col]
    else:
        bench_nav = risky_nav # Self-reference
        
    # 3. Calculate Signal Vector
    if mode == "Fixed Allocation":
        trade_signal = pd.Series(True, index=risky_nav.index)
        basket_ma = pd.Series(0, index=risky_nav.index) # Dummy
        bench_ma = pd.Series(0, index=risky_nav.index)  # Dummy
    
    elif signal_logic == "Hybrid (Entry: Bench / Exit: Basket)":
        raw_signal, bench_ma, basket_ma = run_hybrid_logic(bench_nav, risky_nav, ma_days)
        # Shift 1 day to trade on next open
        trade_signal = raw_signal.shift(1).fillna(True)
        
    else:
        # Standard Trend Following (Self or Bench)
        if signal_logic == "Broad Market (Nifty)":
            sig_source = bench_nav
        else: 
            sig_source = risky_nav
            
        ma_series = sig_source.rolling(window=ma_days, min_periods=1).mean()
        raw_signal = (sig_source > ma_series).shift(1).fillna(True)
        trade_signal = raw_signal
        basket_ma = ma_series # Visualization placeholder
        bench_ma = ma_series  # Visualization placeholder

    # 4. Slice
    mask = (df_full.index.date >= start_date) & (df_full.index.date <= end_date)
    df_slice = df_full.loc[mask]
    
    if df_slice.empty: return None, None, None, None, None, None

    # Slice all components for Debugging
    risky_nav = risky_nav.loc[mask]
    trade_signal = trade_signal.loc[mask]
    bench_nav = bench_nav.loc[mask]
    basket_ma = basket_ma.loc[mask]
    bench_ma = bench_ma.loc[mask]
    
    # 5. Returns
    risky_ret = risky_nav.pct_change().fillna(0)
    safe_ret = df_slice[safe_asset].pct_change().fillna(0)
    
    final_ret = np.where(trade_signal == True, risky_ret, safe_ret)
    strat_nav = (1 + final_ret).cumprod() * 100
    strat_series = pd.Series(strat_nav, index=df_slice.index, name="Strategy")
    
    # 6. Create Debug DataFrame
    debug_df = pd.DataFrame({
        'Risky_Basket_Price': risky_nav,
        'Risky_Basket_MA': basket_ma,
        'Benchmark_Price': bench_nav,
        'Benchmark_MA': bench_ma,
        'Signal (1=Eq, 0=Safe)': trade_signal.astype(int),
        'Strategy_NAV': strat_series
    })
    
    return strat_series, df_slice, debug_df, risky_nav, bench_nav, trade_signal

# --- UI ---
st.sidebar.header("1. Upload Data")
f = st.sidebar.file_uploader("Upload", type=['xlsx','csv'])
df_raw = load_data(f)
if df_raw is None: st.stop()
cols = df_raw.columns.tolist()

# --- SETTINGS ---
st.sidebar.header("2. Logic")
mode = st.sidebar.selectbox("Mode", ["Trend Following", "Fixed Allocation"])

def get_weights(label):
    st.sidebar.markdown(f"**{label}**")
    # Allow ALL columns (Equity OR Debt)
    assets = st.sidebar.multiselect("Select Assets", cols, key=label)
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
signal_logic = "Self"
signal_source = None
ma_days = 200

if mode == "Fixed Allocation":
    risky_weights = get_weights("Portfolio")
    rebal_freq = st.sidebar.selectbox("Rebal", ["Daily","Monthly","Yearly"])
else:
    # 1. Basket
    st.sidebar.info("You can now add Debt funds here if you want a 'Balanced' risky basket.")
    risky_weights = get_weights("Risky Basket Construction")
    rebal_freq = st.sidebar.selectbox("Basket Rebal", ["Daily","Monthly","Yearly"], index=1)
    
    # 2. Safe Asset
    st.sidebar.markdown("---")
    d_guess = [c for c in cols if "Money" in c or "Liquid" in c or "Debt" in c]
    safe_asset = st.sidebar.selectbox("Safe Asset (Cash Component)", cols, index=cols.index(d_guess[0]) if d_guess else 0)
    
    # 3. Logic
    st.sidebar.markdown("---")
    signal_logic = st.sidebar.radio("Logic Type", 
                                    ["Hybrid (Entry: Bench / Exit: Basket)", 
                                     "Broad Market (Nifty)", 
                                     "Self (Basket Only)"])
    
    if "Hybrid" in signal_logic or "Broad" in signal_logic:
        n_guess = [c for c in cols if "Nifty" in c and "50" in c]
        signal_source = st.sidebar.selectbox("Select Benchmark (Entry Signal)", cols, index=cols.index(n_guess[0]) if n_guess else 0)
    
    ma_days = st.sidebar.number_input("DMA Period", 200)

st.sidebar.markdown("---")
bench_comp = st.sidebar.selectbox("Comparison Benchmark", cols, index=0)
valid = df_raw.index
s_d = st.sidebar.date_input("Start", max(valid.min().date(), date(2014,1,1)))
e_d = st.sidebar.date_input("End", valid.max().date())

# --- RUN ---
if not risky_weights or sum(risky_weights.values()) != 1.0: st.stop()

strat, df_s, debug_df, r_nav, b_nav, sig_vec = run_engine(
    df_raw, mode, risky_weights, safe_asset, ma_days, rebal_freq, s_d, e_d, signal_logic, signal_source
)

if strat is None: st.error("No Data"); st.stop()

bench = df_s[bench_comp]
bench = (bench/bench.iloc[0])*100
final = pd.DataFrame({'Strategy': strat, 'Benchmark': bench})

# --- METRICS ---
st.title("Strategy Audit Dashboard")

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

# --- CHART ---
st.plotly_chart(px.line(final, title="Growth of 100"), use_container_width=True)

# --- DEBUG & AUDIT TAB ---
t1, t2, t3 = st.tabs(["🚦 Signal Check", "📋 Calculation Audit", "📊 Monthly Returns"])

with t1:
    st.write(f"**Logic Mode:** {signal_logic}")
    if "Hybrid" in signal_logic:
        st.info("Hybrid Rule: Enter if Benchmark > Bench MA. Exit if Basket < Basket MA.")
        
        # Dual Chart
        fig = go.Figure()
        # Basket (Exit Driver)
        fig.add_trace(go.Scatter(x=debug_df.index, y=debug_df['Risky_Basket_Price'], name="Basket Price", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=debug_df.index, y=debug_df['Risky_Basket_MA'], name="Basket MA (Exit Level)", line=dict(color='orange', dash='dot')))
        
        # Bench (Entry Driver) - Scaled to fit graph
        scale_factor = debug_df['Risky_Basket_Price'].iloc[0] / debug_df['Benchmark_Price'].iloc[0]
        scaled_bench = debug_df['Benchmark_Price'] * scale_factor
        scaled_bench_ma = debug_df['Benchmark_MA'] * scale_factor
        
        fig.add_trace(go.Scatter(x=debug_df.index, y=scaled_bench, name="Benchmark (Scaled)", line=dict(color='gray', width=1)))
        fig.add_trace(go.Scatter(x=debug_df.index, y=scaled_bench_ma, name="Bench MA (Entry Level)", line=dict(color='red', width=1, dash='dot')))
        
        # Highlight Cash Zones
        cash_mask = debug_df['Signal (1=Eq, 0=Safe)'] == 0
        cash_df = debug_df[cash_mask]
        if not cash_df.empty:
            fig.add_trace(go.Scatter(x=cash_df.index, y=[debug_df['Risky_Basket_Price'].min()]*len(cash_df), 
                                     mode='markers', name="In Cash", marker=dict(color='red', symbol='square')))
            
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(debug_df[['Risky_Basket_Price', 'Risky_Basket_MA']])

with t2:
    st.markdown("### Daily Calculation Log")
    st.markdown("Download this to verify exactly why the strategy bought or sold on any specific day.")
    st.dataframe(debug_df.style.format("{:.2f}"), use_container_width=True)
    
    csv = debug_df.to_csv().encode('utf-8')
    st.download_button("Download CSV Calculation", csv, "strategy_audit.csv", "text/csv")

with t3:
    m = final['Strategy'].resample('ME').apply(lambda x: x.iloc[-1]/x.iloc[0]-1 if len(x)>0 else 0)
    p = pd.DataFrame({'Y': m.index.year, 'M': m.index.strftime('%b'), 'V': m.values}).pivot(index='Y', columns='M', values='V')
    p = p.reindex(columns=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    p['YTD'] = final['Strategy'].resample('YE').apply(lambda x: x.iloc[-1]/x.iloc[0]-1).values
    st.dataframe(p.style.format("{:.2%}", na_rep="-").background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)