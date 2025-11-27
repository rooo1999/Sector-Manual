import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import date

st.set_page_config(page_title="Regime Filter Strategy", layout="wide")

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
    """ Creates a synthetic index (NAV) for the Risky Basket. """
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

def run_strategy_engine(df_full, mode, risky_weights, safe_asset, ma_days, rebal_freq, start_date, end_date, signal_source_col=None):
    
    # 1. Construct The Investment Vehicle (The Risky Basket)
    risky_nav = construct_basket(df_full, risky_weights, rebal_freq)
    
    # 2. Determine The Signal Generator
    if mode == "Trend Following" and signal_source_col:
        # Use Nifty 50 (or selected benchmark) as the Signal
        signal_series = df_full[signal_source_col]
    else:
        # Use the Basket itself as the Signal (Self-Trend)
        signal_series = risky_nav

    # 3. Calculate Signals
    # We calculate MA on the FULL history to minimize cold-start issues
    ma_series = signal_series.rolling(window=ma_days).mean()
    
    # Logic: Price > MA = Risk On (True)
    # Shift(1) is CRITICAL to avoid lookahead bias (decision made on yesterday's close)
    raw_signal = (signal_series > ma_series).shift(1)
    
    # --- FIX FOR FIRST 200 DAYS ---
    # User Request: "Keep strategy on risk on for first 200 days"
    # We fill NaNs (which occur before MA exists) with TRUE.
    raw_signal = raw_signal.fillna(True)
    
    # 4. Slice Data to User Selection
    mask = (df_full.index.date >= start_date) & (df_full.index.date <= end_date)
    df_slice = df_full.loc[mask]
    
    if df_slice.empty: return None, None

    # Slice the pre-calculated series to match view
    risky_nav = risky_nav.loc[mask]
    raw_signal = raw_signal.loc[mask]
    signal_series = signal_series.loc[mask]
    ma_series = ma_series.loc[mask]
    
    # Rebase visual NAVs
    risky_nav = (risky_nav / risky_nav.iloc[0]) * 100
    
    # 5. Calculate Returns
    risky_ret = risky_nav.pct_change().fillna(0)
    safe_ret = df_slice[safe_asset].pct_change().fillna(0)
    
    if mode == "Fixed Allocation":
        final_ret = risky_ret 
    else:
        # Vectorized Switch: If Signal True -> Equity, Else -> Debt
        final_ret = np.where(raw_signal == True, risky_ret, safe_ret)
        
    strat_nav = (1 + final_ret).cumprod() * 100
    strat_series = pd.Series(strat_nav, index=df_slice.index, name="Strategy")
    
    return strat_series, df_slice, raw_signal, signal_series, ma_series

# --- UI SETUP ---
st.sidebar.header("1. Upload Data")
f = st.sidebar.file_uploader("Upload Excel/CSV", type=['xlsx','csv'])
df_raw = load_data(f)

if df_raw is None: 
    st.info(f"Waiting for data... (Default: {DEFAULT_PATH})")
    st.stop()

cols = df_raw.columns.tolist()

# --- CONFIGURATION ---
st.sidebar.header("2. Strategy Settings")
mode = st.sidebar.selectbox("Strategy Mode", ["Trend Following", "Fixed Allocation"])

# Helper
def get_weights(label):
    st.sidebar.markdown(f"**{label}**")
    assets = st.sidebar.multiselect(f"Select Assets", cols, key=label)
    w_dict = {}
    if assets:
        def_w = 100/len(assets)
        total = 0
        cols_ui = st.sidebar.columns(1)
        for a in assets:
            val = st.sidebar.number_input(f"{a} %", 0, 100, int(def_w), key=f"w_{a}_{label}")
            w_dict[a] = val/100.0
            total += val
        if total != 100: st.sidebar.error("Weights must sum to 100%")
    return w_dict

risky_weights = {}
safe_asset = cols[0]
ma_days = 200
rebal_freq = "Monthly"
signal_source = None

if mode == "Fixed Allocation":
    risky_weights = get_weights("Portfolio Composition")
    rebal_freq = st.sidebar.selectbox("Rebalancing", ["Daily","Monthly","Yearly","Never"])
else:
    # 1. Basket
    risky_weights = get_weights("Step 1: Risky Basket (What we Buy)")
    rebal_freq = st.sidebar.selectbox("Basket Rebalancing", ["Daily","Monthly","Yearly","Never"], index=1)
    
    # 2. Safe Asset
    st.sidebar.markdown("---")
    d_guess = [c for c in cols if "Money" in c or "Liquid" in c or "Debt" in c]
    safe_asset = st.sidebar.selectbox("Step 2: Safe Asset (Where we Hide)", cols, index=cols.index(d_guess[0]) if d_guess else 0)
    
    # 3. Signal
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Step 3: The Traffic Light (Signal)**")
    signal_type = st.sidebar.radio("Generate Signals based on:", ["Self-Trend (Basket's own MA)", "Broad Market (Nifty 50 MA)"])
    
    if signal_type == "Broad Market (Nifty 50 MA)":
        n_guess = [c for c in cols if "Nifty" in c and "50" in c]
        signal_source = st.sidebar.selectbox("Select Broad Market Index", cols, index=cols.index(n_guess[0]) if n_guess else 0)
        st.sidebar.info(f"💡 Logic: If {signal_source} > 200 DMA, buy Risky Basket. Else, sit in {safe_asset}.")
    else:
        st.sidebar.info("💡 Logic: If Risky Basket > 200 DMA, stay Invested. Else, sit in Safe Asset.")
        
    ma_days = st.sidebar.number_input("DMA Period", value=200)

# Benchmark & Dates
st.sidebar.markdown("---")
bench_col = st.sidebar.selectbox("Benchmark for Comparison", cols, index=0)
valid_dates = df_raw.index
s_date = st.sidebar.date_input("Start Date", max(valid_dates.min().date(), date(2014,1,1)))
e_date = st.sidebar.date_input("End Date", valid_dates.max().date())

# --- EXECUTION ---
if not risky_weights or sum(risky_weights.values()) != 1.0:
    st.warning("⚠️ Weights must sum to 100%.")
    st.stop()

strat_series, df_sliced, signal_vec, sig_price, sig_ma = run_strategy_engine(
    df_raw, mode, risky_weights, safe_asset, ma_days, rebal_freq, s_date, e_date, signal_source
)

if strat_series is None:
    st.error("No data available.")
    st.stop()

# Align Benchmark
bench_series = df_sliced[bench_col]
bench_series = (bench_series / bench_series.iloc[0]) * 100
final_df = pd.DataFrame({'Strategy': strat_series, 'Benchmark': bench_series})

# --- DISPLAY ---
st.title("Strategy Analytics")

# Stats Calc
def get_stats(s):
    if s.empty: return 0,0,0,0
    y = (s.index[-1]-s.index[0]).days/365.25
    cagr = (s.iloc[-1]/s.iloc[0])**(1/y)-1 if y>0 else 0
    vol = s.pct_change().dropna().std()*np.sqrt(252)
    dd = ((s/s.cummax())-1).min()
    return cagr, vol, dd, s.iloc[-1]

sc, sv, sd, send = get_stats(final_df['Strategy'])
bc, bv, bd, bend = get_stats(final_df['Benchmark'])

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("CAGR", f"{sc:.2%}", f"{(sc-bc)*100:.2f} pts")
c2.metric("Max Drawdown", f"{sd:.2%}", f"{(sd-bd)*100:.2f} pts", delta_color="inverse")
c3.metric("Volatility", f"{sv:.2%}", f"{(sv-bv)*100:.2f} pts", delta_color="inverse")
c4.metric("Win Rate vs Bench", "N/A", "Coming Soon")

# Chart
st.subheader("Performance")
fig = px.line(final_df, title="Growth of 100")
st.plotly_chart(fig, use_container_width=True)

# Signal Chart (Debug)
if mode == "Trend Following":
    with st.expander("🚦 View Traffic Light Signals (Trend Source)"):
        sig_df = pd.DataFrame({'Price': sig_price, '200 DMA': sig_ma})
        # Scale to 100 for viz if using raw index
        sig_df = (sig_df / sig_df.iloc[0]) * 100 
        
        fig_sig = px.line(sig_df, title=f"Signal Source Trend ({signal_source if signal_source else 'Basket'})")
        # Overlay Green/Red zones
        st.plotly_chart(fig_sig, use_container_width=True)
        
        st.caption("When Blue Line (Price) is ABOVE Red Line (DMA), Strategy is in EQUITIES. Otherwise CASH.")

# Table & Heatmap
t1, t2 = st.tabs(["Monthly Heatmap", "Yearly Returns"])
with t1:
    m = final_df['Strategy'].resample('ME').apply(lambda x: x.iloc[-1]/x.iloc[0]-1 if len(x)>0 else 0)
    p = pd.DataFrame({'Y': m.index.year, 'M': m.index.strftime('%b'), 'V': m.values}).pivot(index='Y', columns='M', values='V')
    p = p.reindex(columns=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    p['YTD'] = final_df['Strategy'].resample('YE').apply(lambda x: x.iloc[-1]/x.iloc[0]-1).values
    st.dataframe(p.style.format("{:.2%}", na_rep="-").background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)

with t2:
    y = final_df.resample('YE').last()
    start_row = pd.DataFrame([100, 100], index=['Strategy','Benchmark'], columns=[final_df.index[0]-pd.Timedelta(days=1)]).T
    calc = pd.concat([start_row, y]).sort_index()
    y_ret = calc.pct_change().dropna()
    y_ret['Alpha'] = y_ret['Strategy'] - y_ret['Benchmark']
    y_ret.index = y_ret.index.year
    st.dataframe(y_ret.style.format("{:.2%}").background_gradient(cmap='RdYlGn', subset=['Strategy','Alpha']), use_container_width=True)