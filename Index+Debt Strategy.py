import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import date

st.set_page_config(page_title="Hybrid Strategy Engine", layout="wide")

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

def calculate_hybrid_signal(bench_series, basket_series, ma_days):
    """
    Hybrid Logic:
    - Entry: If Bench > Bench_MA
    - Exit:  If Basket < Basket_MA
    """
    # 1. Calculate MAs
    bench_ma = bench_series.rolling(window=ma_days, min_periods=1).mean()
    basket_ma = basket_series.rolling(window=ma_days, min_periods=1).mean()
    
    # Values for loop
    b_price = bench_series.values
    b_ma = bench_ma.values
    k_price = basket_series.values # k for basket
    k_ma = basket_ma.values
    
    signals = []
    # Default State: 1 (Invested) for the start
    state = 1 
    
    for i in range(len(b_price)):
        # Safety for NaNs
        if np.isnan(b_ma[i]) or np.isnan(k_ma[i]):
            signals.append(True)
            continue
            
        if state == 1:
            # We are Invested. Check EXIT condition (Basket based)
            # Exit if Basket < Basket MA
            if k_price[i] < k_ma[i]:
                state = 0
        else:
            # We are in Cash. Check ENTRY condition (Benchmark based)
            # Enter if Benchmark > Benchmark MA
            if b_price[i] > b_ma[i]:
                state = 1
                
        signals.append(bool(state))
        
    return pd.Series(signals, index=bench_series.index)

def run_strategy_engine(df_full, mode, risky_weights, safe_asset, ma_days, rebal_freq, 
                       start_date, end_date, signal_source_col, signal_logic_type):
    
    # 1. Risky Basket
    risky_nav = construct_basket(df_full, risky_weights, rebal_freq)
    
    # 2. Benchmark Series (for Hybrid logic)
    if signal_source_col:
        bench_nav = df_full[signal_source_col]
    else:
        bench_nav = risky_nav # Fallback
        
    # 3. Generate Signals
    if mode == "Fixed Allocation":
        # Always True
        trade_signal = pd.Series(True, index=risky_nav.index)
    
    elif signal_logic_type == "Hybrid (Entry: Benchmark / Exit: Basket)":
        # --- NEW HYBRID LOGIC ---
        raw_signal = calculate_hybrid_signal(bench_nav, risky_nav, ma_days)
        # Shift 1 day to trade tomorrow
        trade_signal = raw_signal.shift(1).fillna(True)
        
    else:
        # --- STANDARD LOGIC (Single Source) ---
        if signal_logic_type == "Broad Market (Nifty)":
            sig_source = bench_nav
        else: 
            sig_source = risky_nav
            
        ma_series = sig_source.rolling(window=ma_days, min_periods=1).mean()
        raw_signal = (sig_source > ma_series).shift(1).fillna(True)
        trade_signal = raw_signal

    # 4. Slice Data
    mask = (df_full.index.date >= start_date) & (df_full.index.date <= end_date)
    df_slice = df_full.loc[mask]
    
    if df_slice.empty: return None, None, None, None, None

    risky_nav = risky_nav.loc[mask]
    trade_signal = trade_signal.loc[mask]
    
    # 5. Returns
    risky_ret = risky_nav.pct_change().fillna(0)
    safe_ret = df_slice[safe_asset].pct_change().fillna(0)
    
    final_ret = np.where(trade_signal == True, risky_ret, safe_ret)
        
    strat_nav = (1 + final_ret).cumprod() * 100
    strat_series = pd.Series(strat_nav, index=df_slice.index, name="Strategy")
    
    return strat_series, df_slice, trade_signal, risky_nav, bench_nav

# --- UI SETUP ---
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
signal_logic_type = "Self"
signal_source_col = None
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
    
    # New Hybrid Option Added
    signal_logic_type = st.sidebar.radio("Logic Type", 
                                         ["Hybrid (Entry: Benchmark / Exit: Basket)", 
                                          "Broad Market (Nifty)", 
                                          "Self (Basket Only)"])
    
    if "Benchmark" in signal_logic_type or "Broad Market" in signal_logic_type:
        n_guess = [c for c in cols if "Nifty" in c and "50" in c]
        signal_source_col = st.sidebar.selectbox("Select Benchmark Index", cols, index=cols.index(n_guess[0]) if n_guess else 0)
    
    ma_days = st.sidebar.number_input("DMA Period", value=200)

# Date
st.sidebar.markdown("---")
bench_comp_col = st.sidebar.selectbox("Comparison Benchmark", cols, index=0)
valid = df_raw.index
s_d = st.sidebar.date_input("Start", max(valid.min().date(), date(2014,1,1)))
e_d = st.sidebar.date_input("End", valid.max().date())

# --- RUN ---
if not risky_weights or sum(risky_weights.values()) != 1.0: st.stop()

strat, df_s, sig_vec, risky_nav_viz, bench_nav_viz = run_strategy_engine(
    df_raw, mode, risky_weights, safe_asset, ma_days, rebal_freq, s_d, e_d, signal_source_col, signal_logic_type
)

if strat is None: st.error("No Data"); st.stop()

bench = df_s[bench_comp_col]
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

# --- VISUALS ---
if mode == "Trend Following":
    with st.expander("🚦 Signal Diagnostic (Logic Visualizer)", expanded=True):
        st.write(f"**Logic Used:** {signal_logic_type}")
        
        # Calculate MAs for Viz
        basket_ma = risky_nav_viz.rolling(ma_days).mean()
        
        if signal_logic_type == "Hybrid (Entry: Benchmark / Exit: Basket)":
             st.info("Blue Line = Basket (Used for Exit). Orange Line = Benchmark (Used for Entry).")
             # Normalize both to start at 100 for comparison
             b_norm = (risky_nav_viz / risky_nav_viz.iloc[0]) * 100
             n_norm = (bench_nav_viz / bench_nav_viz.iloc[0]) * 100
             
             fig_hyb = go.Figure()
             fig_hyb.add_trace(go.Scatter(x=b_norm.index, y=b_norm, name="Basket (Strategy)"))
             fig_hyb.add_trace(go.Scatter(x=n_norm.index, y=n_norm, name="Benchmark (Entry Signal)"))
             
             # Overlay Signal State
             # Identify where Signal is 0 (Cash)
             cash_zones = sig_vec[sig_vec == False]
             fig_hyb.add_trace(go.Scatter(x=cash_zones.index, y=[b_norm.min()]*len(cash_zones), 
                                          mode='markers', name="In Cash", marker=dict(color='red', symbol='square')))
             
             st.plotly_chart(fig_hyb, use_container_width=True)
             
        else:
            # Standard Plot
            fig_sig = go.Figure()
            fig_sig.add_trace(go.Scatter(x=risky_nav_viz.index, y=risky_nav_viz, name="Price"))
            fig_sig.add_trace(go.Scatter(x=basket_ma.index, y=basket_ma, name=f"{ma_days} DMA"))
            st.plotly_chart(fig_sig, use_container_width=True)

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