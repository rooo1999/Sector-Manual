import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import date

st.set_page_config(page_title="Strategy Showdown", layout="wide")

# --- 1. ROBUST DATA LOADER ---
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

# --- 2. BASKET CONSTRUCTION (RISKY ASSET) ---
def get_basket_nav(df, weights, rebal_freq):
    assets = list(weights.keys())
    returns = df[assets].pct_change().fillna(0)
    
    vals = np.array([100.0 * weights[a] for a in assets])
    hist = [100.0]
    dates = [returns.index[0]]
    ret_arr = returns.values
    target_w = np.array([weights[a] for a in assets])
    idx = returns.index
    
    is_never = "Never" in rebal_freq
    
    for i in range(1, len(idx)):
        vals = vals * (1 + ret_arr[i])
        total = np.sum(vals)
        
        reb = False
        if rebal_freq == "Daily": reb = True
        elif rebal_freq == "Monthly" and idx[i].month != idx[i-1].month: reb = True
        elif rebal_freq == "Yearly" and idx[i].year != idx[i-1].year: reb = True
        
        if reb and not is_never:
            vals = total * target_w
            
        hist.append(total)
        dates.append(idx[i])
        
    return pd.Series(hist, index=dates, name="Basket")

# --- 3. STRATEGY ENGINES ---

def run_strategy(df_slice, risky_nav, safe_col, strategy_type):
    """
    Runs the specific logic and returns the NAV series.
    """
    risky_ret = risky_nav.pct_change().fillna(0)
    safe_ret = df_slice[safe_col].pct_change().fillna(0)
    
    # Logic Container
    signal = pd.Series(False, index=risky_nav.index)
    
    if strategy_type == "Buy & Hold (No Switching)":
        signal[:] = True # Always invested
        
    elif strategy_type == "Simple Trend (Price > 200 DMA)":
        # Classic Logic
        ma_200 = risky_nav.rolling(200, min_periods=1).mean()
        # Buy if Price > 200 DMA
        signal = (risky_nav > ma_200).shift(1).fillna(True)
        
    elif strategy_type == "Golden Cross (50 > 200 DMA)":
        # Smoother Logic
        ma_50 = risky_nav.rolling(50, min_periods=1).mean()
        ma_200 = risky_nav.rolling(200, min_periods=1).mean()
        # Buy if 50 SMA > 200 SMA
        signal = (ma_50 > ma_200).shift(1).fillna(True)

    # Calculate Returns
    # np.where(True, Equity, Debt)
    final_ret = np.where(signal, risky_ret, safe_ret)
    
    nav = (1 + final_ret).cumprod() * 100
    return pd.Series(nav, index=risky_nav.index, name=strategy_type)

# --- 4. UI & EXECUTION ---
st.sidebar.header("1. Data Input")
f = st.sidebar.file_uploader("Upload Data", type=['xlsx','csv'])
df_raw = load_data(f)
if df_raw is None: st.info(f"Load Data... {DEFAULT_PATH}"); st.stop()
cols = df_raw.columns.tolist()

st.sidebar.header("2. Asset Config")
# Risky Basket
st.sidebar.markdown("**Risky Basket (Equity)**")
assets = st.sidebar.multiselect("Select Assets", cols, default=cols[:2] if len(cols)>1 else cols)
weights = {}
if assets:
    def_w = 100/len(assets)
    for a in assets:
        weights[a] = st.sidebar.number_input(f"{a}%",0,100,int(def_w))/100.0
basket_freq = st.sidebar.selectbox("Rebalance Frequency", ["Monthly", "Yearly", "Never"])

# Safe Asset
st.sidebar.markdown("**Safe Asset (Debt)**")
d_guess = [c for c in cols if "Money" in c or "Liquid" in c or "Debt" in c]
safe_asset = st.sidebar.selectbox("Select Fund", cols, index=cols.index(d_guess[0]) if d_guess else 0)

# Date
st.sidebar.markdown("---")
valid = df_raw.index
s_d = st.sidebar.date_input("Start", max(valid.min().date(), date(2014,1,1)))
e_d = st.sidebar.date_input("End", valid.max().date())

# --- 5. RUN ANALYSIS ---
# Slice Data
mask = (df_raw.index.date >= s_d) & (df_raw.index.date <= e_d)
df_slice = df_raw.loc[mask]
# Build Risky NAV first (needs to be sliced to match)
# Actually better to build on FULL data then slice to allow MA calculation
basket_full = get_basket_nav(df_raw, weights, basket_freq)
basket_slice = basket_full.loc[mask]

# Rebase Basket to 100 at start of period
basket_slice = (basket_slice / basket_slice.iloc[0]) * 100

# Run Strategies
results = pd.DataFrame(index=basket_slice.index)

# 1. Benchmark (Buy & Hold the Basket)
results["Buy & Hold (Basket)"] = run_strategy(df_slice, basket_slice, safe_asset, "Buy & Hold (No Switching)")

# 2. Simple Trend
results["Simple Trend (200 DMA)"] = run_strategy(df_slice, basket_slice, safe_asset, "Simple Trend (Price > 200 DMA)")

# 3. Golden Cross
results["Golden Cross (50 > 200)"] = run_strategy(df_slice, basket_slice, safe_asset, "Golden Cross (50 > 200 DMA)")

# --- 6. DISPLAY DASHBOARD ---
st.title("Strategy Showdown: Which Logic Wins?")

# Metrics Calculation
metrics = []
for col in results.columns:
    s = results[col]
    
    # CAGR
    days = (s.index[-1] - s.index[0]).days
    cagr = (s.iloc[-1]/s.iloc[0])**(365.25/days) - 1 if days > 0 else 0
    
    # Volatility
    vol = s.pct_change().dropna().std() * np.sqrt(252)
    
    # Drawdown
    dd = ((s/s.cummax()) - 1).min()
    
    # Negative Months %
    m_ret = s.resample('ME').last().pct_change().dropna()
    neg_months = (m_ret < 0).sum()
    total_months = len(m_ret)
    neg_pct = neg_months / total_months if total_months > 0 else 0
    
    # Sort Key (Sharpe-ish)
    score = cagr / abs(dd) if dd != 0 else 0
    
    metrics.append({
        "Strategy": col,
        "CAGR": cagr,
        "Max Drawdown": dd,
        "Volatility": vol,
        "% Negative Months": neg_pct,
        "Score": score
    })

metrics_df = pd.DataFrame(metrics).set_index("Strategy")
metrics_df = metrics_df.sort_values("Score", ascending=False) # Best first

# Top Winner Card
winner = metrics_df.index[0]
st.success(f"🏆 **Winner:** {winner} (Best Risk-Adjusted Return)")

# Main Chart
st.plotly_chart(px.line(results, title="Growth of 100 (Direct Comparison)"), use_container_width=True)

# Comparison Table
st.subheader("In-Depth Analysis Table")
st.table(metrics_df.style.format({
    "CAGR": "{:.2%}",
    "Max Drawdown": "{:.2%}", 
    "Volatility": "{:.2%}",
    "% Negative Months": "{:.1%}",
    "Score": "{:.2f}"
}).background_gradient(cmap="RdYlGn", subset=["CAGR", "Score"])
  .background_gradient(cmap="RdYlGn_r", subset=["Max Drawdown", "Volatility", "% Negative Months"]))

# Monthly Returns for the Winner
st.subheader(f"Monthly Returns: {winner}")
s_win = results[winner]
m_ret = s_win.resample('ME').apply(lambda x: x.iloc[-1]/x.iloc[0]-1 if len(x)>0 else 0)
piv = pd.DataFrame({'Y': m_ret.index.year, 'M': m_ret.index.strftime('%b'), 'V': m_ret.values})
piv = piv.pivot(index='Y', columns='M', values='V')
piv = piv.reindex(columns=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
piv['YTD'] = s_win.resample('YE').apply(lambda x: x.iloc[-1]/x.iloc[0]-1).values
st.dataframe(piv.style.format("{:.2%}", na_rep="-").background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)