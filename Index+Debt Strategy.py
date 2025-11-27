import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import date

st.set_page_config(page_title="Trend Strategy Pro", layout="wide")

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

# --- CORE CALCULATIONS ---

def construct_basket(df_full, weights_dict, rebalance_freq):
    assets = list(weights_dict.keys())
    returns = df_full[assets].pct_change().fillna(0)
    
    vals = np.array([100.0 * weights_dict[a] for a in assets])
    hist = [100.0]
    dates = [returns.index[0]]
    ret_arr = returns.values
    target_w = np.array([weights_dict[a] for a in assets])
    idx = returns.index
    
    is_never = "Never" in rebalance_freq
    
    for i in range(1, len(idx)):
        vals = vals * (1 + ret_arr[i])
        total = np.sum(vals)
        
        reb = False
        if rebalance_freq == "Daily": reb = True
        elif rebalance_freq == "Monthly" and idx[i].month != idx[i-1].month: reb = True
        elif rebalance_freq == "Yearly" and idx[i].year != idx[i-1].year: reb = True
        
        if reb and not is_never:
            vals = total * target_w
            
        hist.append(total)
        dates.append(idx[i])
        
    return pd.Series(hist, index=dates, name="Basket_NAV")

def get_ma(series, window, ma_type):
    """ Calculates Moving Average (SMA or EMA). """
    if ma_type == "SMA":
        return series.rolling(window=window, min_periods=1).mean()
    else:
        return series.ewm(span=window, adjust=False, min_periods=1).mean()

def run_hybrid_logic(bench_daily, basket_daily, ma_period, ma_type, frequency):
    """
    Calculates signals based on Resampled Data (Weekly/Monthly) 
    to filter noise and map it back to daily data.
    """
    # 1. Resample Data
    if frequency == "Weekly":
        # Resample to Friday (W-FRI)
        bench_res = bench_daily.resample('W-FRI').last()
        basket_res = basket_daily.resample('W-FRI').last()
    elif frequency == "Monthly":
        # Resample to Month End
        bench_res = bench_daily.resample('ME').last()
        basket_res = basket_daily.resample('ME').last()
    else:
        # Daily
        bench_res = bench_daily
        basket_res = basket_daily

    # 2. Calculate MA on Resampled Data
    bench_ma_res = get_ma(bench_res, ma_period, ma_type)
    basket_ma_res = get_ma(basket_res, ma_period, ma_type)
    
    # 3. Generate Signals on Resampled Data
    b_price = bench_res.values
    b_ma = bench_ma_res.values
    k_price = basket_res.values
    k_ma = basket_ma_res.values
    
    signals = []
    state = 1 # Start Invested
    
    for i in range(len(b_price)):
        if np.isnan(b_ma[i]) or np.isnan(k_ma[i]):
            signals.append(True)
            continue
            
        if state == 1:
            # EXIT Rule (Basket)
            if k_price[i] < k_ma[i]:
                state = 0
        else:
            # ENTRY Rule (Benchmark)
            if b_price[i] > b_ma[i]:
                state = 1
        
        signals.append(bool(state))
        
    signal_res = pd.Series(signals, index=bench_res.index)
    
    # 4. Map back to Daily (Forward Fill)
    # This applies Friday's signal to next Monday-Friday
    signal_daily = signal_res.reindex(bench_daily.index, method='ffill').fillna(True)
    
    # For visualization, we need daily MAs (interpolated or ffilled for visual comparison)
    # Actually, simpler to just calculate Daily MA for the chart even if logic used Weekly
    # But to be accurate to logic, we should visualize the stepped MA.
    bench_ma_daily = bench_ma_res.reindex(bench_daily.index, method='ffill')
    basket_ma_daily = basket_ma_res.reindex(basket_daily.index, method='ffill')
    
    return signal_daily, bench_ma_daily, basket_ma_daily

def run_engine(df_full, mode, risky_weights, safe_asset, ma_period, ma_type, rebal_freq, 
               start_date, end_date, signal_logic, signal_source_col, sig_frequency):
    
    # 1. Build Risky Basket
    risky_nav = construct_basket(df_full, risky_weights, rebal_freq)
    
    # 2. Benchmark
    if signal_source_col:
        bench_nav = df_full[signal_source_col]
    else:
        bench_nav = risky_nav 
        
    # 3. Calculate Signals
    if mode == "Fixed Allocation":
        trade_signal = pd.Series(True, index=risky_nav.index)
        basket_ma = pd.Series(0, index=risky_nav.index)
        bench_ma = pd.Series(0, index=risky_nav.index)
    
    elif signal_logic == "Hybrid (Entry: Bench / Exit: Basket)":
        raw_signal, bench_ma, basket_ma = run_hybrid_logic(bench_nav, risky_nav, ma_period, ma_type, sig_frequency)
        # Shift 1 day (Signal calc at close -> Trade at next open)
        trade_signal = raw_signal.shift(1).fillna(True)
        
    else:
        # Standard Trend Following
        if signal_logic == "Broad Market (Nifty)":
            sig_source = bench_nav
        else: 
            sig_source = risky_nav
        
        # Resample logic for single source
        if sig_frequency == "Weekly":
            sig_res = sig_source.resample('W-FRI').last()
        elif sig_frequency == "Monthly":
            sig_res = sig_source.resample('ME').last()
        else:
            sig_res = sig_source
            
        ma_res = get_ma(sig_res, ma_period, ma_type)
        raw_sig_res = (sig_res > ma_res)
        
        # Map back
        trade_signal = raw_sig_res.reindex(sig_source.index, method='ffill').shift(1).fillna(True)
        
        # Viz data
        basket_ma = ma_res.reindex(sig_source.index, method='ffill')
        bench_ma = basket_ma 

    # 4. Slice
    mask = (df_full.index.date >= start_date) & (df_full.index.date <= end_date)
    df_slice = df_full.loc[mask]
    
    if df_slice.empty: return None, None, None, None, None, None

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
    
    # 6. Debug
    debug_df = pd.DataFrame({
        'Basket_Price': risky_nav,
        'Basket_MA_Ref': basket_ma,
        'Bench_Price': bench_nav,
        'Bench_MA_Ref': bench_ma,
        'Active_Signal': trade_signal.astype(int)
    })
    
    return strat_series, df_slice, debug_df, risky_nav, bench_nav, trade_signal

# --- UI ---
st.sidebar.header("1. Upload Data")
f = st.sidebar.file_uploader("Upload", type=['xlsx','csv'])
df_raw = load_data(f)
if df_raw is None: st.stop()
cols = df_raw.columns.tolist()

# --- SETTINGS ---
st.sidebar.header("2. Strategy Logic")
mode = st.sidebar.selectbox("Mode", ["Trend Following", "Fixed Allocation"])

def get_weights(label):
    st.sidebar.markdown(f"**{label}**")
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
ma_period = 40 # Default changed for weekly
sig_freq = "Weekly"
ma_type = "SMA"

if mode == "Fixed Allocation":
    risky_weights = get_weights("Portfolio")
    rebal_freq = st.sidebar.selectbox("Rebal", ["Daily","Monthly","Yearly","Never (Buy & Hold)"])
else:
    # 1. Basket
    risky_weights = get_weights("Risky Basket Construction")
    rebal_freq = st.sidebar.selectbox("Basket Rebal", ["Daily","Monthly","Yearly","Never (Buy & Hold)"])
    
    # 2. Safe Asset
    st.sidebar.markdown("---")
    d_guess = [c for c in cols if "Money" in c or "Liquid" in c or "Debt" in c]
    safe_asset = st.sidebar.selectbox("Safe Asset", cols, index=cols.index(d_guess[0]) if d_guess else 0)
    
    # 3. Logic
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Trend Logic**")
    signal_logic = st.sidebar.radio("Method", 
                                    ["Hybrid (Entry: Bench / Exit: Basket)", 
                                     "Broad Market (Nifty)", 
                                     "Self (Basket Only)"])
    
    if "Hybrid" in signal_logic or "Broad" in signal_logic:
        n_guess = [c for c in cols if "Nifty" in c and "50" in c]
        signal_source = st.sidebar.selectbox("Benchmark Index", cols, index=cols.index(n_guess[0]) if n_guess else 0)
    
    # 4. Parameters
    c1, c2, c3 = st.sidebar.columns(3)
    sig_freq = c1.selectbox("Resolution", ["Daily", "Weekly", "Monthly"], index=1)
    ma_type = c2.selectbox("MA Type", ["SMA", "EMA"])
    
    def_ma = 200 if sig_freq == "Daily" else (40 if sig_freq == "Weekly" else 10)
    ma_period = c3.number_input(f"Period ({sig_freq})", value=def_ma)
    
    st.sidebar.caption(f"Calculating {ma_period}-{sig_freq} {ma_type}.")

st.sidebar.markdown("---")
bench_comp = st.sidebar.selectbox("Benchmark for Comparison", cols, index=0)
valid = df_raw.index
s_d = st.sidebar.date_input("Start", max(valid.min().date(), date(2014,1,1)))
e_d = st.sidebar.date_input("End", valid.max().date())

# --- RUN ---
if not risky_weights or sum(risky_weights.values()) != 1.0: st.stop()

strat, df_s, debug_df, r_nav, b_nav, sig_vec = run_engine(
    df_raw, mode, risky_weights, safe_asset, ma_period, ma_type, rebal_freq, s_d, e_d, signal_logic, signal_source, sig_freq
)

if strat is None: st.error("No Data"); st.stop()

bench = df_s[bench_comp]
bench = (bench/bench.iloc[0])*100
final = pd.DataFrame({'Strategy': strat, 'Benchmark': bench})

# --- METRICS ---
st.title(f"Trend Strategy ({sig_freq} Signal)")

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

# Row 1: Strategy
c1, c2, c3 = st.columns(3)
c1.metric("Strategy CAGR", f"{sc:.2%}", f"{(sc-bc)*100:.2f} pts")
c2.metric("Strategy Drawdown", f"{sd:.2%}", f"{(sd-bd)*100:.2f} pts", delta_color="inverse")
c3.metric("Strategy Volatility", f"{sv:.2%}", f"{(sv-bv)*100:.2f} pts", delta_color="inverse")

# Row 2: Benchmark
c4, c5, c6 = st.columns(3)
c4.metric(f"{bench_comp} CAGR", f"{bc:.2%}")
c5.metric(f"{bench_comp} Drawdown", f"{bd:.2%}")
c6.metric(f"{bench_comp} Volatility", f"{bv:.2%}")

# --- CHART ---
st.plotly_chart(px.line(final, title="Growth of 100"), use_container_width=True)

# --- DEBUG & AUDIT ---
t1, t2, t3 = st.tabs(["Signal Check", "Calculation Audit", "Yearly/Monthly Returns"])

with t1:
    if "Hybrid" in signal_logic:
        st.info("Blue = Basket (Exit Check). Orange = Benchmark (Entry Check). Steps = Weekly/Monthly MA Levels.")
        fig = go.Figure()
        
        # Scale for viz
        scale = debug_df['Basket_Price'].iloc[0] / debug_df['Bench_Price'].iloc[0]
        
        # Basket (Exit)
        fig.add_trace(go.Scatter(x=debug_df.index, y=debug_df['Basket_Price'], name="Basket Price", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=debug_df.index, y=debug_df['Basket_MA_Ref'], name="Basket MA (Exit)", line=dict(color='blue', dash='dot')))
        
        # Bench (Entry)
        fig.add_trace(go.Scatter(x=debug_df.index, y=debug_df['Bench_Price']*scale, name="Bench Price (Scaled)", line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=debug_df.index, y=debug_df['Bench_MA_Ref']*scale, name="Bench MA (Entry)", line=dict(color='orange', dash='dot')))
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(debug_df[['Basket_Price', 'Basket_MA_Ref']])

with t2:
    st.dataframe(debug_df.style.format("{:.2f}"), use_container_width=True)
    csv = debug_df.to_csv().encode('utf-8')
    st.download_button("Download CSV", csv, "audit.csv", "text/csv")

with t3:
    y = final.resample('YE').last()
    s_r = pd.DataFrame([100, 100], index=['Strategy','Benchmark'], columns=[final.index[0]-pd.Timedelta(days=1)]).T
    c = pd.concat([s_r, y]).sort_index()
    yr = c.pct_change().dropna()
    yr['Alpha'] = yr['Strategy'] - yr['Benchmark']
    yr.index = yr.index.year
    st.dataframe(yr.style.format("{:.2%}").background_gradient(cmap='RdYlGn', subset=['Strategy']), use_container_width=True)