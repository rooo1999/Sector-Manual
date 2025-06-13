import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Dynamic Momentum Strategy Backtester",
    layout="wide",
)

# --- Calculation Functions ---

@st.cache_data
def load_data(file):
    """Loads and preprocesses data from an Excel file."""
    try:
        df = pd.read_excel(file, index_col='Date', parse_dates=True)
        # Ensure all data is numeric, converting errors to NaN, then forward-filling
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.ffill()
        return df
    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
        return None

def calculate_rsi(series, period=14):
    """Calculates RSI using Exponential Moving Average."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, slow=26, fast=12, signal=9):
    """Calculates MACD, Signal Line, and Histogram."""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calculate_performance_metrics(returns_series, risk_free_rate=0.0):
    """Calculates key performance metrics."""
    if returns_series.empty or returns_series.isnull().all():
        return {k: 0 for k in ["Final Value", "CAGR (%)", "Annualized Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)"]}

    initial_capital = 10000
    equity_curve = (1 + returns_series).cumprod() * initial_capital

    total_days = len(returns_series)
    years = total_days / 252.0
    cagr = (equity_curve.iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    volatility = returns_series.std() * np.sqrt(252)
    sharpe_ratio = (cagr - risk_free_rate) / volatility if volatility != 0 else 0
    
    cum_max = equity_curve.cummax()
    drawdown = (equity_curve - cum_max) / cum_max
    max_drawdown = drawdown.min()

    return {
        "Final Value": f"₹{equity_curve.iloc[-1]:,.2f}",
        "CAGR (%)": f"{cagr * 100:.2f}",
        "Annualized Volatility (%)": f"{volatility * 100:.2f}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown (%)": f"{max_drawdown * 100:.2f}"
    }

def run_backtest(price_df, universe_cols, benchmark_col, lookback_period, top_n, cash_rule_params, rsi_params, macd_params):
    """The core backtest engine with dynamic universe and advanced filters."""
    # --- 1. Pre-calculate all necessary data ---
    momentum = price_df.pct_change(lookback_period)
    
    # Technical Indicators
    rsi_df = pd.DataFrame({col: calculate_rsi(price_df[col]) for col in universe_cols})
    macd_data = {col: calculate_macd(price_df[col]) for col in universe_cols}
    macd_line_df = pd.DataFrame({col: data[0] for col, data in macd_data.items()})
    signal_line_df = pd.DataFrame({col: data[1] for col, data in macd_data.items()})
    
    # Data for cash rule
    benchmark_sma = price_df[benchmark_col].rolling(window=cash_rule_params['sma_period']).mean()

    # --- 2. Generate Holdings Signals ---
    rebalance_dates = price_df.resample('M').first().index
    positions = pd.DataFrame(index=price_df.index, columns=universe_cols).fillna(0)
    last_positions = pd.Series(0, index=universe_cols)

    for date in price_df.index:
        if date in rebalance_dates:
            # DYNAMIC UNIVERSE: On this rebalancing day, which sectors have enough data?
            valid_universe = momentum.loc[date].dropna().index.intersection(universe_cols)
            if not valid_universe.any(): # Skip if no sectors are valid yet
                positions.loc[date] = last_positions
                continue

            # RANKING: Rank only the valid sectors
            ranks = momentum.loc[date, valid_universe].rank(ascending=False)
            top_performers = ranks[ranks <= top_n].index

            # ADVANCED FILTERS: Apply secondary confirmation filters
            filtered_top_performers = []
            for sector in top_performers:
                passes_filters = True
                # RSI Filter
                if rsi_params['use']:
                    if rsi_df.loc[date, sector] < rsi_params['min_rsi']:
                        passes_filters = False
                # MACD Filter
                if macd_params['use']:
                    if macd_line_df.loc[date, sector] < signal_line_df.loc[date, sector]:
                        passes_filters = False
                
                if passes_filters:
                    filtered_top_performers.append(sector)
            
            # CASH RULE: Decide if we should be invested at all
            invest_decision = True
            if cash_rule_params['use']:
                benchmark_mom = momentum.loc[date, benchmark_col]
                benchmark_price = price_df.loc[date, benchmark_col]
                benchmark_sma_val = benchmark_sma.loc[date]
                if pd.notna(benchmark_mom) and pd.notna(benchmark_sma_val):
                    if benchmark_mom <= 0 and benchmark_price < benchmark_sma_val:
                        invest_decision = False
            
            # FINAL POSITIONS for this period
            if invest_decision and filtered_top_performers:
                current_positions = pd.Series(0, index=universe_cols)
                current_positions[filtered_top_performers] = 1
            else:
                current_positions = pd.Series(0, index=universe_cols) # Go to cash
            
            last_positions = current_positions
        
        positions.loc[date] = last_positions

    # --- 3. Calculate Returns ---
    daily_returns = price_df.pct_change()
    shifted_positions = positions.shift(1) # Trade on next day's open
    
    # Align data for calculation
    common_index = daily_returns.index.intersection(shifted_positions.index)
    daily_returns, shifted_positions = daily_returns.loc[common_index], shifted_positions.loc[common_index]

    portfolio_daily_returns = (daily_returns[universe_cols] * shifted_positions[universe_cols]).sum(axis=1)
    num_positions = shifted_positions.sum(axis=1).replace(0, 1) # Avoid division by zero
    portfolio_daily_returns /= num_positions

    benchmark_daily_returns = daily_returns.loc[portfolio_daily_returns.index, benchmark_col]

    return portfolio_daily_returns, benchmark_daily_returns, momentum, positions

# --- Streamlit UI ---
st.title("📈 Dynamic Momentum Strategy Backtester")

# --- Sidebar ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your XLSX data file", type="xlsx")

if uploaded_file:
    full_data = load_data(uploaded_file)
    if full_data is not None:
        st.sidebar.header("2. Configure Backtest")
        
        all_cols = full_data.columns.tolist()
        benchmark_col = st.sidebar.selectbox("Select Benchmark", all_cols, index=len(all_cols)-1)
        
        available_universe = [col for col in all_cols if col != benchmark_col]
        universe_cols = st.sidebar.multiselect("Select Trading Universe", available_universe, default=available_universe)

        start_date = st.sidebar.date_input("Start Date", full_data.index.min())
        end_date = st.sidebar.date_input("End Date", full_data.index.max())
        
        # Filter data based on date range
        data_filtered = full_data.loc[start_date:end_date]

        st.sidebar.header("3. Strategy Parameters")
        lookback_period = st.sidebar.slider("Momentum Lookback (days)", 5, 252, 21, 1, help="How far back to look for price momentum.")
        top_n = st.sidebar.slider("Sectors to Hold", 1, max(1, len(universe_cols)), min(2, len(universe_cols)), 1)
        
        st.sidebar.subheader("Filters & Rules")
        # Cash Rule
        use_cash_rule = st.sidebar.checkbox("Use Hybrid Cash Rule?", value=True, help="Hold cash if benchmark trend is negative (short & long term).")
        sma_period = st.sidebar.slider("Long-term SMA for Cash Rule", 50, 252, 210, disabled=not use_cash_rule)
        
        # Advanced Filters
        with st.sidebar.expander("Advanced Indicator Filters (Optional)"):
            use_rsi_filter = st.checkbox("Use RSI Filter?", value=False, help="Only buy if momentum is confirmed by strong RSI.")
            min_rsi = st.slider("Minimum RSI to Enter", 1, 99, 55, disabled=not use_rsi_filter)
            use_macd_filter = st.checkbox("Use MACD Filter?", value=False, help="Only buy if MACD line is above its signal line.")

        if st.sidebar.button("🚀 Run Backtest", type="primary"):
            cash_rule_params = {'use': use_cash_rule, 'sma_period': sma_period}
            rsi_params = {'use': use_rsi_filter, 'min_rsi': min_rsi}
            macd_params = {'use': use_macd_filter}

            strategy_returns, bench_returns, momentum, positions = run_backtest(
                data_filtered, universe_cols, benchmark_col, lookback_period, top_n,
                cash_rule_params, rsi_params, macd_params
            )

            # --- Main Page Display ---
            st.header("📊 Performance Dashboard")

            # Performance Metrics Table
            metrics_data = {
                "Strategy": calculate_performance_metrics(strategy_returns),
                "Benchmark": calculate_performance_metrics(bench_returns)
            }
            metrics_df = pd.DataFrame(metrics_data).T
            st.dataframe(metrics_df, use_container_width=True)

            # Equity Curve Chart
            strategy_eq = (1 + strategy_returns).cumprod() * 10000
            bench_eq = (1 + bench_returns).cumprod() * 10000
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=strategy_eq.index, y=strategy_eq, mode='lines', name='Strategy', line=dict(color='royalblue', width=2)))
            fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq, mode='lines', name='Benchmark', line=dict(color='grey', width=2, dash='dash')))
            fig.update_layout(title='Strategy vs. Benchmark Equity Curve', yaxis_title='Portfolio Value (Log Scale)', yaxis_type="log", legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabs for detailed analysis
            tab1, tab2 = st.tabs(["🗓️ Monthly Returns Analysis", "🔎 Historical Holdings"])

            with tab1:
                st.header("Monthly Returns & Outperformance (%)")
                monthly_data = pd.DataFrame({
                    'Strategy': strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1),
                    'Benchmark': bench_returns.resample('M').apply(lambda x: (1 + x).prod() - 1),
                })
                monthly_data['Outperformance'] = monthly_data['Strategy'] - monthly_data['Benchmark']
                monthly_data.index = monthly_data.index.strftime('%Y-%b')
                st.dataframe((monthly_data * 100).style.format("{:.2f}").background_gradient(cmap='RdYlGn', subset=['Outperformance']), use_container_width=True)

            with tab2:
                st.header("Portfolio Allocations on Rebalancing Dates")
                rebalance_dates = positions.resample('M').first().index
                historical_holdings = positions[positions.index.isin(rebalance_dates) & (positions.diff().abs().sum(axis=1) > 0)]
                display_holdings = historical_holdings.apply(lambda row: ', '.join(row[row==1].index) or 'CASH', axis=1).to_frame("Sectors Held")
                st.dataframe(display_holdings.sort_index(ascending=False), use_container_width=True)
else:
    st.info("Upload an XLSX file using the sidebar to begin.")