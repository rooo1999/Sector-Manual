import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Sector Rotation Strategy Backtester",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Functions for Performance ---
@st.cache_data
def load_data(file):
    """Loads and preprocesses the data from a CSV file."""
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True) # Use dayfirst=True for DD-MM-YYYY
    df = df.set_index('Date')
    # Ensure all data is numeric, forward-fill any missing values
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.ffill().dropna()
    return df

# --- Manual Calculation Functions ---

def calculate_momentum(df, lookback_period):
    """
    MANUAL Calculation of momentum (percentage change).
    Avoids using df.pct_change() to show the manual logic.
    """
    # Shift the dataframe by the lookback period to get the price 'n' days ago
    shifted_df = df.shift(lookback_period)
    # Momentum = (Current Price - Shifted Price) / Shifted Price
    momentum = (df - shifted_df) / shifted_df
    return momentum.dropna()

def get_rankings(momentum_df):
    """
    Ranks the assets based on their momentum scores for each day.
    Rank 1 is the best performing asset.
    """
    # axis=1 ranks across the columns (sectors) for each row (date)
    # ascending=False makes higher momentum get a lower rank number (e.g., Rank 1)
    return momentum_df.rank(axis=1, ascending=False, method='first')

def run_backtest(price_df, benchmark_col, lookback_period, top_n, use_cash_rule):
    """
    The core function to run the entire backtest and generate results.
    """
    # 1. Calculate Momentum and Ranks
    momentum = calculate_momentum(price_df, lookback_period)
    ranks = get_rankings(momentum)

    # Align dataframes to the same date range (starting after the first lookback period)
    aligned_prices = price_df.loc[ranks.index]
    
    # 2. Generate Trading Signals (Holdings)
    positions = pd.DataFrame(index=ranks.index, columns=price_df.columns).fillna(0)
    
    # Identify rebalancing dates (first trading day of each month)
    rebalance_dates = ranks.resample('M').first().index

    last_positions = pd.Series(0, index=price_df.columns)

    for date in ranks.index:
        if date in rebalance_dates:
            # --- Rebalancing Logic ---
            current_ranks = ranks.loc[date]
            
            # --- Cash Rule Application ---
            invest__decision = True # Assume we invest by default
            if use_cash_rule:
                benchmark_momentum = momentum.loc[date, benchmark_col]
                if benchmark_momentum <= 0:
                    invest_decision = False

            if invest_decision:
                # Identify which sectors are in the top N
                top_performers = current_ranks[current_ranks <= top_n].index
                current_positions = pd.Series(0, index=price_df.columns)
                current_positions[top_performers] = 1 # We hold these
            else:
                # Cash Rule triggered: hold no positions
                current_positions = pd.Series(0, index=price_df.columns)

            last_positions = current_positions
        
        positions.loc[date] = last_positions

    # 3. Calculate Portfolio Returns
    # Shift positions by 1 day because we trade based on previous day's signal
    shifted_positions = positions.shift(1).dropna()
    
    # Calculate daily returns of all assets
    daily_returns = price_df.pct_change().dropna()
    
    # Align all dataframes
    common_index = shifted_positions.index.intersection(daily_returns.index)
    shifted_positions = shifted_positions.loc[common_index]
    daily_returns = daily_returns.loc[common_index]

    # Calculate strategy daily return
    # The return is the sum of returns of the assets we hold
    portfolio_daily_returns = (daily_returns * shifted_positions).sum(axis=1)
    
    # Divide by the number of positions held to get the average return
    num_positions = shifted_positions.sum(axis=1)
    portfolio_daily_returns = portfolio_daily_returns / num_positions.replace(0, 1) # Avoid division by zero on cash days

    # 4. Calculate Cumulative Returns (Equity Curve)
    initial_capital = 10000
    portfolio_cumulative = (1 + portfolio_daily_returns).cumprod() * initial_capital
    
    # Benchmark performance
    benchmark_returns = daily_returns[benchmark_col]
    benchmark_cumulative = (1 + benchmark_returns).cumprod() * initial_capital
    
    # Add a starting point to the series for charting
    start_date = portfolio_cumulative.index[0] - pd.Timedelta(days=1)
    portfolio_cumulative[start_date] = initial_capital
    benchmark_cumulative[start_date] = initial_capital
    portfolio_cumulative = portfolio_cumulative.sort_index()
    benchmark_cumulative = benchmark_cumulative.sort_index()

    return portfolio_cumulative, benchmark_cumulative, momentum, ranks, positions, portfolio_daily_returns

# --- Streamlit UI ---

st.title("📈 Sector Rotation Momentum Strategy Backtester")

# --- Sidebar for User Inputs ---
st.sidebar.header("Strategy Parameters")
uploaded_file = st.sidebar.file_uploader("Upload your CSV data file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    
    # Get columns except for the benchmark
    sector_cols = [col for col in data.columns]
    benchmark_col = st.sidebar.selectbox("Select Benchmark Column", data.columns, index=len(data.columns)-1)
    
    # User-defined parameters
    lookback_period = st.sidebar.number_input("Momentum Lookback Period (days)", min_value=5, max_value=252, value=21, step=1)
    top_n = st.sidebar.number_input("Number of Top Sectors to Hold", min_value=1, max_value=10, value=2, step=1)
    use_cash_rule = st.sidebar.radio("Use Cash Rule (Absolute Momentum Filter)?", ("Yes", "No")) == "Yes"
    
    st.sidebar.markdown("""
    **Strategy Rules:**
    1.  **Rebalance:** First trading day of the month.
    2.  **Selection:** Ranks sectors by `N`-day momentum.
    3.  **Entry:** Invests in the `Top N` ranked sectors.
    4.  **Cash Rule (if 'Yes'):** Only invests if the benchmark's own momentum is positive. Otherwise, holds cash.
    """)

    # --- Run Strategy & Display Results ---
    if st.sidebar.button("🚀 Run Backtest"):
        # Run the core logic
        strategy_eq, bench_eq, momentum, ranks, positions, strategy_returns = run_backtest(
            data, benchmark_col, lookback_period, top_n, use_cash_rule
        )

        # --- Tabbed Interface ---
        tab1, tab2, tab3 = st.tabs(["📊 Backtest Performance", "📋 Current & Historical Holdings", "🔢 Detailed Returns"])

        with tab1:
            st.header("Performance Chart")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=strategy_eq.index, y=strategy_eq, mode='lines', name='Strategy', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq, mode='lines', name='Benchmark', line=dict(color='grey', width=2, dash='dash')))
            fig.update_layout(
                title='Strategy vs. Benchmark Equity Curve',
                xaxis_title='Date',
                yaxis_title='Portfolio Value',
                legend_title='Legend',
                yaxis_tickprefix='₹',
                font=dict(family="Arial, sans-serif", size=12)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Performance Metrics
            st.header("Key Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Strategy Final Value", f"₹{strategy_eq.iloc[-1]:,.2f}")
            col2.metric("Benchmark Final Value", f"₹{bench_eq.iloc[-1]:,.2f}")
            
            # Max Drawdown Calculation
            def calculate_max_drawdown(series):
                cum_max = series.cummax()
                drawdown = (series - cum_max) / cum_max
                return drawdown.min() * 100

            strategy_dd = calculate_max_drawdown(strategy_eq)
            bench_dd = calculate_max_drawdown(bench_eq)

            col3.metric("Strategy Max Drawdown", f"{strategy_dd:.2f}%")
            # You can add more metrics like CAGR, Sharpe Ratio etc.

        with tab2:
            st.header(f"Current Sector Momentum ({lookback_period}-day)")
            st.write(f"As of {momentum.index[-1].strftime('%d-%b-%Y')}")
            
            current_snapshot = pd.DataFrame({
                'Momentum (%)': momentum.iloc[-1] * 100,
                'Rank': ranks.iloc[-1]
            }).sort_values('Rank')
            st.dataframe(current_snapshot.style.format({'Momentum (%)': '{:.2f}'}), use_container_width=True)
            
            st.header("Historical Portfolio Allocations")
            st.write("Showing the portfolio composition on each rebalancing date.")
            
            rebalance_dates = positions.resample('M').first().index
            # Get changes in positions on rebalance dates
            historical_holdings = positions[positions.index.isin(rebalance_dates) & (positions.diff().abs().sum(axis=1) > 0)]
            
            # Format for display
            display_holdings = historical_holdings.apply(lambda row: ', '.join(row[row==1].index), axis=1).reset_index()
            display_holdings.columns = ['Rebalance Date', 'Sectors Held']
            display_holdings['Sectors Held'] = display_holdings['Sectors Held'].replace('', 'CASH')
            
            st.dataframe(display_holdings.set_index('Rebalance Date'), use_container_width=True)

        with tab3:
            st.header("Strategy Monthly Returns")
            monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
            monthly_returns_df = monthly_returns.to_frame(name="Return").reset_index()
            monthly_returns_df['Year'] = monthly_returns_df['Date'].dt.year
            monthly_returns_df['Month'] = monthly_returns_df['Date'].dt.strftime('%b')
            
            monthly_pivot = monthly_returns_df.pivot_table(index='Year', columns='Month', values='Return', aggfunc='sum')
            # Order months correctly
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_pivot = monthly_pivot.reindex(columns=month_order)
            
            st.dataframe(monthly_pivot.style.format("{:.2f}%").background_gradient(cmap='RdYlGn', axis=None).highlight_null(''), use_container_width=True)

else:
    st.info("Please upload a CSV file to begin.")
    st.markdown("""
    **Instructions:**
    1.  Prepare your data in Excel or Google Sheets.
    2.  Column A should be 'Date' in `DD-MM-YYYY` or `MM-DD-YYYY` format.
    3.  Subsequent columns should contain the price data for each sector index (e.g., 'Nifty Auto', 'Nifty Bank').
    4.  The last column should be your benchmark (e.g., 'Nifty 50').
    5.  Save the file as a CSV.
    6.  Upload it using the sidebar on the left.
    7.  Set your desired parameters and click 'Run Backtest'.
    """)