import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Sector Rotation Backtester",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching & Data Loading ---
@st.cache_data
def load_data(file):
    """Loads and preprocesses data from an Excel file."""
    try:
        df = pd.read_excel(file)
        if 'Date' not in df.columns:
            st.error("Error: A 'Date' column was not found in the uploaded file.")
            return None
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.set_index('Date')
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.ffill() # Forward-fill to handle intermittent missing data
        return df
    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
        return None

# --- Calculation Functions ---

def calculate_performance_metrics(returns_series, risk_free_rate=0.0):
    """Calculates key performance metrics from a daily returns series."""
    if returns_series.empty:
        return {
            "Final Value": 10000, "CAGR (%)": 0, "Annualized Volatility (%)": 0,
            "Sharpe Ratio": 0, "Max Drawdown (%)": 0
        }
    
    # Cumulative return and final value
    initial_capital = 10000
    equity_curve = (1 + returns_series).cumprod() * initial_capital

    # CAGR
    total_days = len(returns_series)
    years = total_days / 252  # Approximate trading days in a year
    cagr = (equity_curve.iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 else 0

    # Annualized Volatility
    volatility = returns_series.std() * np.sqrt(252)

    # Sharpe Ratio
    sharpe_ratio = (cagr - risk_free_rate) / volatility if volatility != 0 else 0

    # Max Drawdown
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

def run_backtest(price_df, benchmark_col, lookback_period, top_n, use_cash_rule, sma_period):
    """The core function to run the entire backtest."""
    # 1. Calculate Momentum and Ranks for the given universe
    momentum = price_df.pct_change(lookback_period).dropna()
    ranks = momentum.rank(axis=1, ascending=False, method='first')

    # 2. Calculate Long-term SMA for the benchmark (for our hybrid cash rule)
    benchmark_sma = price_df[benchmark_col].rolling(window=sma_period).mean()

    # Align all dataframes to a common start date
    common_index = momentum.index.intersection(benchmark_sma.index)
    momentum, ranks, benchmark_sma = momentum.loc[common_index], ranks.loc[common_index], benchmark_sma.loc[common_index]

    # 3. Generate Holdings Signals
    positions = pd.DataFrame(index=ranks.index, columns=price_df.columns).fillna(0)
    rebalance_dates = ranks.resample('M').first().index
    last_positions = pd.Series(0, index=price_df.columns)

    for date in ranks.index:
        if date in rebalance_dates:
            current_ranks = ranks.loc[date]
            invest_decision = True
            
            if use_cash_rule:
                benchmark_momentum_value = momentum.loc[date, benchmark_col]
                benchmark_price = price_df.loc[date, benchmark_col]
                benchmark_sma_value = benchmark_sma.loc[date]

                # HYBRID CASH RULE: Go to cash only if BOTH short and long term trends are negative
                if benchmark_momentum_value <= 0 and benchmark_price < benchmark_sma_value:
                    invest_decision = False
            
            if invest_decision:
                top_performers = current_ranks[current_ranks <= top_n].index
                current_positions = pd.Series(0, index=price_df.columns)
                current_positions[top_performers] = 1
            else:
                current_positions = pd.Series(0, index=price_df.columns) # Cash position
            
            last_positions = current_positions
        
        positions.loc[date] = last_positions

    # 4. Calculate Portfolio Returns
    shifted_positions = positions.shift(1).dropna()
    daily_returns = price_df.pct_change().dropna()
    
    common_index = shifted_positions.index.intersection(daily_returns.index)
    shifted_positions, daily_returns = shifted_positions.loc[common_index], daily_returns.loc[common_index]

    portfolio_daily_returns = (daily_returns * shifted_positions).sum(axis=1)
    num_positions = shifted_positions.sum(axis=1)
    portfolio_daily_returns = portfolio_daily_returns / num_positions.replace(0, 1)

    benchmark_daily_returns = daily_returns.loc[portfolio_daily_returns.index, benchmark_col]

    return portfolio_daily_returns, benchmark_daily_returns, momentum, ranks, positions

# --- Streamlit UI ---

st.title("🚀 Advanced Sector Rotation Momentum Backtester")

# --- Sidebar ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your XLSX data file", type="xlsx")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    
    if data is not None:
        st.sidebar.header("2. Set Backtest Period")
        start_date = st.sidebar.date_input("Start Date", data.index.min(), min_value=data.index.min(), max_value=data.index.max())
        end_date = st.sidebar.date_input("End Date", data.index.max(), min_value=data.index.min(), max_value=data.index.max())

        if start_date > end_date:
            st.sidebar.error("Error: Start date must be before end date.")
        else:
            # Filter data based on selected dates and drop any columns that have NaNs in this period
            data_filtered = data.loc[start_date:end_date].dropna(axis=1)
            st.sidebar.info(f"Using {len(data_filtered.columns)} sectors with full data in the selected period.")

            st.sidebar.header("3. Configure Strategy")
            benchmark_col = st.sidebar.selectbox("Select Benchmark Column", data_filtered.columns, index=len(data_filtered.columns)-1)
            lookback_period = st.sidebar.slider("Momentum Lookback (days)", 5, 252, 21, 1)
            top_n = st.sidebar.slider("Number of Sectors to Hold", 1, max(1, len(data_filtered.columns)-1), 2, 1)
            
            use_cash_rule = st.sidebar.checkbox("Use Hybrid Cash Rule?", value=True)
            sma_period = st.sidebar.slider("Long-term SMA for Cash Rule (days)", 50, 252, 210, 10, disabled=not use_cash_rule)
            
            st.sidebar.markdown("""
            ---
            **Hybrid Cash Rule:** If checked, the strategy holds cash unless the benchmark's 1-month momentum is positive **OR** its price is above the long-term SMA.
            """)

            if st.sidebar.button("Run Backtest", type="primary"):
                # Run the backtest
                strategy_returns, bench_returns, momentum, ranks, positions = run_backtest(
                    data_filtered, benchmark_col, lookback_period, top_n, use_cash_rule, sma_period
                )

                # --- Main Page Display ---
                st.header("📈 Performance Dashboard")

                # Calculate metrics
                strategy_metrics = calculate_performance_metrics(strategy_returns)
                benchmark_metrics = calculate_performance_metrics(bench_returns)

                metrics_df = pd.DataFrame([strategy_metrics, benchmark_metrics], index=["Strategy", "Benchmark"])
                st.dataframe(metrics_df, use_container_width=True)

                # Equity Curve
                strategy_eq = (1 + strategy_returns).cumprod() * 10000
                bench_eq = (1 + bench_returns).cumprod() * 10000
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=strategy_eq.index, y=strategy_eq, mode='lines', name='Strategy', line=dict(color='royalblue', width=2)))
                fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq, mode='lines', name='Benchmark', line=dict(color='grey', width=2, dash='dash')))
                fig.update_layout(title='Strategy vs. Benchmark Equity Curve', yaxis_title='Portfolio Value (Log Scale)', yaxis_type="log")
                st.plotly_chart(fig, use_container_width=True)
                
                # --- Tabbed Interface for Details ---
                tab1, tab2, tab3 = st.tabs(["Monthly Returns", "Current Standings", "Historical Holdings"])

                with tab1:
                    st.header("Monthly Returns Comparison (%)")
                    # Strategy
                    strat_monthly = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                    # Benchmark
                    bench_monthly = bench_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                    
                    monthly_df = pd.DataFrame({
                        'Strategy': strat_monthly * 100,
                        'Benchmark': bench_monthly * 100,
                    })
                    monthly_df['Outperformance'] = monthly_df['Strategy'] - monthly_df['Benchmark']
                    monthly_df.index = monthly_df.index.strftime('%Y-%b')

                    st.dataframe(monthly_df.style.format("{:.2f}").background_gradient(
                        cmap='RdYlGn', subset=['Strategy', 'Benchmark', 'Outperformance']
                    ), use_container_width=True)


                with tab2:
                    st.header(f"Current Sector Standings ({lookback_period}-day)")
                    st.write(f"As of {momentum.index[-1].strftime('%d-%b-%Y')}")
                    
                    current_snapshot = pd.DataFrame({
                        'Momentum (%)': momentum.iloc[-1] * 100,
                        'Rank': ranks.iloc[-1]
                    }).sort_values('Rank')
                    st.dataframe(current_snapshot.style.format({'Momentum (%)': '{:.2f}'}), use_container_width=True)

                with tab3:
                    st.header("Historical Portfolio Allocations")
                    st.write("Showing the portfolio composition on each rebalancing date.")
                    rebalance_dates = positions.resample('M').first().index
                    historical_holdings = positions[positions.index.isin(rebalance_dates) & (positions.diff().abs().sum(axis=1) > 0)]
                    
                    display_holdings = historical_holdings.apply(lambda row: ', '.join(row[row==1].index) or 'CASH', axis=1).reset_index()
                    display_holdings.columns = ['Rebalance Date', 'Sectors Held']
                    
                    st.dataframe(display_holdings.set_index('Rebalance Date').sort_index(ascending=False), use_container_width=True)

else:
    st.info("Awaiting upload of an XLSX file...")
    st.image("https://i.imgur.com/gYf0g39.png", width=600)
    st.markdown("""
    **Welcome to the Advanced Sector Rotation Backtester!**
    1.  Prepare your data in an Excel file (`.xlsx`).
    2.  The first column must be named `Date`.
    3.  Subsequent columns for sector/index price data.
    4.  Upload via the sidebar to begin.
    """)