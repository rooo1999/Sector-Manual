import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="Momentum Sector Trading Strategy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Indicator Calculation Functions (No pandas_ta) ---

def calculate_momentum(series, n_days):
    """Calculates n-day momentum."""
    # Using .iloc[-1] and .iloc[-n_days-1] to avoid issues with date alignment
    if len(series) < n_days + 1:
        return np.nan
    return (series.iloc[-1] / series.iloc[-n_days - 1]) - 1

def calculate_sma(series, window):
    """Calculates Simple Moving Average."""
    if len(series) < window:
        return np.nan
    return series.rolling(window=window).mean().iloc[-1]

def calculate_rsi(series, window=14):
    """Calculates Relative Strength Index (RSI)."""
    if len(series) < window + 1:
        return np.nan
    
    delta = series.diff()
    gain = delta.where(delta > 0, 0).dropna()
    loss = -delta.where(delta < 0, 0).dropna()

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Handle division by zero if avg_loss is 0
    if avg_loss.iloc[-1] == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    """Calculates Moving Average Convergence Divergence (MACD)."""
    if len(series) < slow_period:
        return np.nan, np.nan
    
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    return macd_line.iloc[-1], signal_line.iloc[-1]

def calculate_volatility(series, window=21):
    """Calculates annualised volatility."""
    if len(series) < window:
        return np.nan
    daily_returns = series.pct_change()
    # 252 trading days in a year
    vol = daily_returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)
    return vol

# --- Performance Metrics ---

def calculate_cagr(series):
    """Calculates Compound Annual Growth Rate."""
    if series.empty or series.iloc[0] == 0:
        return 0
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    num_years = (series.index[-1] - series.index[0]).days / 365.25
    if num_years == 0:
        return 0
    return (end_val / start_val) ** (1 / num_years) - 1

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculates Sharpe Ratio."""
    if returns.empty:
        return 0.0
    # Using 252 trading days
    annualized_return = returns.mean() * 252
    annualized_std = returns.std() * np.sqrt(252)
    if annualized_std == 0:
        return np.nan
    sharpe = (annualized_return - risk_free_rate) / annualized_std
    return sharpe
    
def calculate_max_drawdown(series):
    """Calculates Maximum Drawdown."""
    if series.empty:
        return 0.0
    cumulative_max = series.cummax()
    drawdown = (series - cumulative_max) / cumulative_max
    return drawdown.min()

# --- Main Application ---
def main():
    st.title("📈 Momentum-Based Sector Rotation Strategy")
    st.markdown("""
    This dashboard backtests a sector rotation strategy based on momentum and other technical indicators. 
    The strategy selects the top N sectors each month, holds them for the month, and then rebalances.
    """)

    # --- Sidebar for User Inputs ---
    with st.sidebar:
        st.header("⚙️ Strategy Parameters")

        uploaded_file = st.file_uploader(
            "Upload your Excel data file", 
            type=["xlsx", "xls"],
            help="File should have a 'Date' column, followed by sector/index price columns."
        )

        if uploaded_file:
            @st.cache_data
            def load_data(file):
                df = pd.read_excel(file, parse_dates=['Date'])
                return df
            
            sample_df = load_data(uploaded_file)
            
            # --- Date and Sector Selection ---
            st.subheader("General Settings")
            min_date = sample_df['Date'].min().date()
            max_date = sample_df['Date'].max().date()
            
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
            
            # ## --- CHANGE START: Make benchmark selectable ---
            # Get all columns that are not 'Date'
            potential_benchmark_cols = [col for col in sample_df.columns if col != 'Date']
            if not potential_benchmark_cols:
                st.error("Your file must contain price columns besides the 'Date' column.")
                return

            # Intelligently guess the default benchmark
            default_benchmark_index = 0
            for i, col in enumerate(potential_benchmark_cols):
                if 'bench' in col.lower() or 'nifty' in col.lower() and '50' in col.lower():
                    default_benchmark_index = i
                    break
            
            benchmark_col = st.selectbox(
                "Select Benchmark Column", 
                potential_benchmark_cols, 
                index=default_benchmark_index
            )

            # Define sectors as all columns except Date and the chosen Benchmark
            all_sectors = [col for col in sample_df.columns if col not in ['Date', benchmark_col]]
            if not all_sectors:
                st.warning("No sectors available to trade after selecting the benchmark. Please upload a file with at least one sector column and one benchmark column.")
                return
            # ## --- CHANGE END ---

            num_sectors_to_invest = st.slider(
                "Number of Top Sectors to Select", 
                min_value=1, 
                max_value=len(all_sectors), 
                value=min(2, len(all_sectors)), # Default to 2 or max available
                step=1
            )
            
            # --- Indicator Lookbacks ---
            with st.expander("Indicator Lookback Periods"):
                mom_1m_lookback = st.number_input("1-Month Momentum Lookback (days)", value=21)
                mom_3m_lookback = st.number_input("3-Month Momentum Lookback (days)", value=63)
                mom_6m_lookback = st.number_input("6-Month Momentum Lookback (days)", value=126)
                sma_lookback = st.number_input("SMA Lookback (days)", value=50)
                rsi_lookback = st.number_input("RSI Lookback (days)", value=14)
                volatility_lookback = st.number_input("Volatility Lookback (days)", value=21)

            # --- Indicator Weights ---
            with st.expander("Indicator Ranking Weights"):
                st.markdown("Set the importance of each indicator for ranking. Higher is better.")
                weight_mom_1m = st.slider("1-Month Momentum Weight", 0.0, 1.0, 0.25)
                weight_mom_3m = st.slider("3-Month Momentum Weight", 0.0, 1.0, 0.25)
                weight_mom_6m = st.slider("6-Month Momentum Weight", 0.0, 1.0, 0.20)
                weight_sma_ratio = st.slider("Price/SMA Ratio Weight", 0.0, 1.0, 0.10, help="Ranks sectors with price furthest above their SMA higher.")
                weight_rsi = st.slider("RSI Weight", 0.0, 1.0, 0.10)
                weight_macd = st.slider("MACD Weight", 0.0, 1.0, 0.05)
                weight_volatility = st.slider("Inverse Volatility Weight", 0.0, 1.0, 0.05, help="Ranks lower volatility sectors higher.")
        else:
            st.info("Awaiting for an Excel file to be uploaded.")
            return

    # --- Main Panel: Data Loading and Backtesting ---
    if st.sidebar.button("🚀 Run Backtest"):
        
        # --- 1. Data Preparation ---
        df = sample_df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.loc[str(start_date):str(end_date)]
        df.dropna(axis=0, how='all', inplace=True) # Drop rows where all values are missing
        df.ffill(inplace=True) # Forward-fill to handle missing values like holidays

        sectors = all_sectors
        
        # --- 2. Backtesting Logic ---
        with st.spinner('Running backtest... This might take a moment.'):
            # Get rebalancing dates (first trading day of each month)
            rebalance_dates = df.resample('MS').first().index

            portfolio_returns = []
            historical_selections = {}
            last_month_selections = []

            for i in range(len(rebalance_dates) - 1):
                ranking_date = rebalance_dates[i]
                start_period = rebalance_dates[i]
                end_period = rebalance_dates[i+1]
                
                hist_data = df.loc[:ranking_date]
                if hist_data.empty: continue

                ranks = pd.DataFrame(index=sectors)
                indicator_values = {}
                for sector in sectors:
                    series = hist_data[sector].dropna()
                    indicator_values[sector] = {
                        'mom_1m': calculate_momentum(series, mom_1m_lookback),
                        'mom_3m': calculate_momentum(series, mom_3m_lookback),
                        'mom_6m': calculate_momentum(series, mom_6m_lookback),
                        'sma_ratio': series.iloc[-1] / calculate_sma(series, sma_lookback) if calculate_sma(series, sma_lookback) else np.nan,
                        'rsi': calculate_rsi(series, rsi_lookback),
                        'macd': calculate_macd(series)[0],
                        'volatility': calculate_volatility(series, volatility_lookback)
                    }

                indicator_df = pd.DataFrame(indicator_values).T.dropna()
                
                if indicator_df.empty or len(indicator_df) < num_sectors_to_invest: continue

                ranks['rank_mom_1m'] = indicator_df['mom_1m'].rank(ascending=False)
                ranks['rank_mom_3m'] = indicator_df['mom_3m'].rank(ascending=False)
                ranks['rank_mom_6m'] = indicator_df['mom_6m'].rank(ascending=False)
                ranks['rank_sma_ratio'] = indicator_df['sma_ratio'].rank(ascending=False)
                ranks['rank_rsi'] = indicator_df['rsi'].rank(ascending=False)
                ranks['rank_macd'] = indicator_df['macd'].rank(ascending=False)
                ranks['rank_volatility'] = indicator_df['volatility'].rank(ascending=True)
                
                ranks['composite_score'] = (
                    weight_mom_1m * ranks['rank_mom_1m'] +
                    weight_mom_3m * ranks['rank_mom_3m'] +
                    weight_mom_6m * ranks['rank_mom_6m'] +
                    weight_sma_ratio * ranks['rank_sma_ratio'] +
                    weight_rsi * ranks['rank_rsi'] +
                    weight_macd * ranks['rank_macd'] +
                    weight_volatility * ranks['rank_volatility']
                )
                
                ranks = ranks.dropna()
                top_sectors = ranks.sort_values(by='composite_score', ascending=True).head(num_sectors_to_invest).index.tolist()
                
                historical_selections[start_period.strftime('%Y-%m')] = top_sectors
                
                investment_period_df = df.loc[start_period:end_period]
                daily_returns_selected_sectors = investment_period_df[top_sectors].pct_change().dropna()
                monthly_portfolio_returns = daily_returns_selected_sectors.mean(axis=1)
                portfolio_returns.append(monthly_portfolio_returns)
            
            if not portfolio_returns:
                st.error("Could not generate a portfolio. This might be due to a short date range or insufficient data for the lookback periods.")
                return

            # --- 6. Consolidate and Analyze Results ---
            strategy_returns = pd.concat(portfolio_returns)
            strategy_cumulative = (1 + strategy_returns).cumprod()
            
            # ## --- CHANGE START: Use the selected benchmark column ---
            benchmark_returns = df[benchmark_col].pct_change().loc[strategy_returns.index]
            # ## --- CHANGE END ---
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            
            # --- Performance Metrics Calculation ---
            strategy_cagr = calculate_cagr(strategy_cumulative)
            benchmark_cagr = calculate_cagr(benchmark_cumulative)
            strategy_sharpe = calculate_sharpe_ratio(strategy_returns)
            benchmark_sharpe = calculate_sharpe_ratio(benchmark_returns)
            strategy_mdd = calculate_max_drawdown(strategy_cumulative)
            benchmark_mdd = calculate_max_drawdown(benchmark_cumulative)

            total_turnover = 0
            for month, selections in historical_selections.items():
                if last_month_selections:
                    turnover = len(set(selections) - set(last_month_selections))
                    total_turnover += turnover
                last_month_selections = selections
            churn_ratio = total_turnover / (len(historical_selections) * num_sectors_to_invest) if historical_selections else 0

            # --- 7. Display Results ---
            st.header("📊 Backtest Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Strategy CAGR", f"{strategy_cagr:.2%}")
                st.metric(f"{benchmark_col} CAGR", f"{benchmark_cagr:.2%}")
            with col2:
                st.metric("Strategy Sharpe Ratio", f"{strategy_sharpe:.2f}")
                st.metric(f"{benchmark_col} Sharpe", f"{benchmark_sharpe:.2f}")
            with col3:
                st.metric("Strategy Max Drawdown", f"{strategy_mdd:.2%}")
                st.metric(f"{benchmark_col} Max Drawdown", f"{benchmark_mdd:.2%}")
            with col4:
                st.metric("Strategy Churn Ratio", f"{churn_ratio:.2%}", help="The percentage of the portfolio that is replaced each month on average.")

            st.subheader("Equity Curve: Strategy vs. Benchmark")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=strategy_cumulative.index, y=strategy_cumulative, mode='lines', name='Strategy'))
            fig.add_trace(go.Scatter(x=benchmark_cumulative.index, y=benchmark_cumulative, mode='lines', name=benchmark_col))
            fig.update_layout(title='Portfolio Value Over Time (Rebased to 1)', yaxis_title='Cumulative Growth', xaxis_title='Date')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📜 Historical Data")
            
            with st.expander("View Historical Monthly Sector Selections"):
                selections_df = pd.DataFrame.from_dict(historical_selections, orient='index')
                selections_df.index.name = 'Month'
                selections_df.columns = [f'Sector_{i+1}' for i in range(num_sectors_to_invest)]
                st.dataframe(selections_df)
            
            with st.expander("View Monthly Returns Breakdown"):
                monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                monthly_returns_df = monthly_returns.to_frame(name='Strategy Returns')
                monthly_returns_df.index = monthly_returns_df.index.strftime('%Y-%m')
                st.dataframe(monthly_returns_df.style.format("{:.2%}"))

if __name__ == '__main__':
    main()