import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import date

st.set_page_config(page_title="Strategy Showdown Pro", layout="wide")

# --- 1. DATA ENGINE ---
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

# --- 2. LOGIC ENGINE ---

def get_basket_nav(df, weights):
    assets = list(weights.keys())
    returns = df[assets].pct_change().fillna(0)
    vals = np.array([100.0 * weights[a] for a in assets])
    hist, dates = [100.0], [returns.index[0]]
    ret_arr = returns.values
    target_w = np.array([weights[a] for a in assets])
    
    # Monthly Rebalancing assumed for the basket construction itself
    idx = returns.index
    for i in range(1, len(idx)):
        vals = vals * (1 + ret_arr[i])
        total = np.sum(vals)
        if idx[i].month != idx[i-1].month: # Rebalance monthly
            vals = total * target_w
        hist.append(total)
        dates.append(idx[i])
        
    return pd.Series(hist, index=dates, name="Basket")

def calculate_trade_stats(signal_series, daily_returns):
    """
    Analyzes the 'Signal' (1=Invested, 0=Cash) to find entries/exits and wins/losses.
    """
    # Detect changes in signal
    trades = signal_series.diff().fillna(0)
    entries = trades[trades == 1].index
    exits = trades[trades == -1].index
    
    # If currently invested, add temporary exit at end date for calc
    if signal_series.iloc[-1] == 1:
        exits = exits.append(pd.Index([daily_returns.index[-1]]))
        
    # If started invested, add temporary entry at start
    if signal_series.iloc[0] == 1:
        entries = entries.insert(0, daily_returns.index[0])
        
    # Make equal length
    n_trades = min(len(entries), len(exits))
    entries = entries[:n_trades]
    exits = exits[:n_trades]
    
    trade_returns = []
    for en, ex in zip(entries, exits):
        # Calculate compounded return for this period
        period_ret = daily_returns.loc[en:ex]
        total_ret = (1 + period_ret).prod() - 1
        trade_returns.append(total_ret)
        
    if not trade_returns:
        return 0, 0.0, 0.0
        
    trade_returns = np.array(trade_returns)
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns <= 0]
    
    win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    
    return n_trades, win_rate, avg_win/abs(avg_loss) if avg_loss != 0 else 0

def run_strategy(df_slice, risky_nav, safe_col, strategy_type):
    risky_ret = risky_nav.pct_change().fillna(0)
    safe_ret = df_slice[safe_col].pct_change().fillna(0)
    
    signal = pd.Series(False, index=risky_nav.index)
    
    if "Buy & Hold" in strategy_type:
        signal[:] = True
    elif "Simple Trend" in strategy_type:
        ma_200 = risky_nav.rolling(200, min_periods=1).mean()
        signal = (risky_nav > ma_200).shift(1).fillna(True)
    elif "Golden Cross" in strategy_type:
        ma_50 = risky_nav.rolling(50, min_periods=1).mean()
        ma_200 = risky_nav.rolling(200, min_periods=1).mean()
        signal = (ma_50 > ma_200).shift(1).fillna(True)

    final_ret = np.where(signal, risky_ret, safe_ret)
    nav = (1 + final_ret).cumprod() * 100
    
    # Calculate Trade Stats based on the SIGNAL and the RISKY RETURN (Captured)
    n_trades, win_rate, profit_factor = calculate_trade_stats(signal, risky_ret)
    
    return pd.Series(nav, index=risky_nav.index), n_trades, win_rate, profit_factor

# --- 3. UI ---
st.sidebar.header("1. Input")
f = st.sidebar.file_uploader("Upload Data", type=['xlsx','csv'])
df_raw = load_data(f)
if df_raw is None: st.info(f"Load Data... {DEFAULT_PATH}"); st.stop()
cols = df_raw.columns.tolist()

# Assets
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

st.sidebar.markdown("**Benchmark (Nifty)**")
n_guess = [c for c in cols if "Nifty" in c and "50" in c]
bench_col = st.sidebar.selectbox("Comparison", cols, index=cols.index(n_guess[0]) if n_guess else 0)

# Dates
st.sidebar.markdown("---")
valid = df_raw.index
s_d = st.sidebar.date_input("Start", max(valid.min().date(), date(2014,1,1)))
e_d = st.sidebar.date_input("End", valid.max().date())

# --- 4. EXECUTION ---
mask = (df_raw.index.date >= s_d) & (df_raw.index.date <= e_d)
df_slice = df_raw.loc[mask]

# Basket
basket_full = get_basket_nav(df_raw, weights)
basket_slice = basket_full.loc[mask]
basket_slice = (basket_slice / basket_slice.iloc[0]) * 100

# Benchmark NAV
bench_nav = df_slice[bench_col]
bench_nav = (bench_nav / bench_nav.iloc[0]) * 100

# Run Strategies
results_data = {}

# Nifty 50 (Reference)
results_data["Nifty 50 (Benchmark)"] = {
    "NAV": bench_nav, "Trades": 0, "WinRate": 0, "PF": 0
}

# Buy & Hold Basket
s_bh, t_bh, w_bh, p_bh = run_strategy(df_slice, basket_slice, safe_asset, "Buy & Hold")
results_data["Buy & Hold (Your Basket)"] = {"NAV": s_bh, "Trades": t_bh, "WinRate": w_bh, "PF": p_bh}

# Simple Trend
s_st, t_st, w_st, p_st = run_strategy(df_slice, basket_slice, safe_asset, "Simple Trend (200 DMA)")
results_data["Simple Trend (200 DMA)"] = {"NAV": s_st, "Trades": t_st, "WinRate": w_st, "PF": p_st}

# Golden Cross
s_gc, t_gc, w_gc, p_gc = run_strategy(df_slice, basket_slice, safe_asset, "Golden Cross (50/200)")
results_data["Golden Cross (50/200)"] = {"NAV": s_gc, "Trades": t_gc, "WinRate": w_gc, "PF": p_gc}


# --- 5. ANALYTICS ---
st.title("Strategy Showdown: Beat Nifty 50")

# Compile Metrics Table
metrics_list = []
combined_nav = pd.DataFrame()

for name, data in results_data.items():
    s = data["NAV"]
    combined_nav[name] = s
    
    # CAGR
    days = (s.index[-1] - s.index[0]).days
    cagr = (s.iloc[-1]/s.iloc[0])**(365.25/days) - 1 if days > 0 else 0
    
    # Vol & DD
    vol = s.pct_change().dropna().std() * np.sqrt(252)
    dd = ((s/s.cummax()) - 1).min()
    
    metrics_list.append({
        "Strategy": name,
        "CAGR": cagr,
        "Max Drawdown": dd,
        "Volatility": vol,
        "Trades": data["Trades"],
        "Win Rate": data["WinRate"],
        "Final Value": s.iloc[-1]
    })

met_df = pd.DataFrame(metrics_list).set_index("Strategy")
met_df = met_df.sort_values("CAGR", ascending=False)

# VISUALS
# 1. Main Table
st.subheader("Performance Matrix")
st.dataframe(met_df.style.format({
    "CAGR": "{:.2%}", "Max Drawdown": "{:.2%}", "Volatility": "{:.2%}", 
    "Win Rate": "{:.1%}", "Final Value": "{:.0f}"
}).background_gradient(cmap="RdYlGn", subset=["CAGR", "Win Rate"])
  .background_gradient(cmap="RdYlGn_r", subset=["Max Drawdown", "Volatility"]), 
  use_container_width=True)

# 2. Chart
st.subheader("Growth Comparison")
fig = px.line(combined_nav)
fig.update_traces(line=dict(width=2))
# Make Nifty Bold Black
fig.update_traces(selector=dict(name="Nifty 50 (Benchmark)"), line=dict(color='black', width=4, dash='dot'))
st.plotly_chart(fig, use_container_width=True)

# 3. Yearly Breakdown
st.subheader("Yearly Returns Comparison")
yearly_res = combined_nav.resample('YE').last()
# Handle start
start_row = pd.DataFrame([100]*len(yearly_res.columns), index=yearly_res.columns, columns=[combined_nav.index[0]-pd.Timedelta(days=1)]).T
calc = pd.concat([start_row, yearly_res]).sort_index()
y_ret = calc.pct_change().dropna()
y_ret.index = y_ret.index.year

# Highlight Nifty Column logic
st.dataframe(y_ret.style.format("{:.2%}")
             .background_gradient(cmap="RdYlGn", axis=1), # Compare columns (strategies) against each other per year
             use_container_width=True)

# 4. Detailed Drawdown View
with st.expander("Show Drawdown Chart"):
    dd_df = (combined_nav / combined_nav.cummax()) - 1
    st.plotly_chart(px.area(dd_df, title="Drawdown Depth"), use_container_width=True)