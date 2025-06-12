import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Momentum Sector Strategy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- All calculation and metric functions remain the same as before ---
# (calculate_momentum, calculate_sma, calculate_rsi, calculate_macd, calculate_volatility)
# (calculate_cagr, calculate_sharpe_ratio, calculate_max_drawdown)

def calculate_momentum(series, n_days):
    if len(series) < n_days + 1: return np.nan
    return (series.iloc[-1] / series.iloc[-n_days - 1]) - 1

def calculate_sma(series, window):
    if len(series) < window: return np.nan
    return series.rolling(window=window).mean().iloc[-1]

def calculate_rsi(series, window=14):
    if len(series) < window + 1: return np.nan
    delta = series.diff()
    gain = delta.where(delta > 0, 0).dropna()
    loss = -delta.where(delta < 0, 0).dropna()
    if len(gain) < window or len(loss) < window: return np.nan
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    if avg_loss.iloc[-1] == 0: return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    if len(series) < slow_period: return np.nan, np.nan
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line.iloc[-1], signal_line.iloc[-1]

def calculate_volatility(series, window=21):
    if len(series) < window: return np.nan
    daily_returns = series.pct_change()
    vol = daily_returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)
    return vol

def calculate_cagr(series):
    if series.empty or series.iloc[0] == 0: return 0
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    num_years = (series.index[-1] - series.index[0]).days / 365.25
    if num_years == 0: return 0
    return (end_val / start_val) ** (1 / num_years) - 1

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    if returns.empty: return 0.0
    annualized_return = returns.mean() * 252
    annualized_std = returns.std() * np.sqrt(252)
    if annualized_std == 0: return np.nan
    sharpe = (annualized_return - risk_free_rate) / annualized_std
    return sharpe
    
def calculate_max_drawdown(series):
    if series.empty: return 0.0
    cumulative_max = series.cummax()
    drawdown = (series - cumulative_max) / cumulative_max
    return drawdown.min()

# --- Main Application ---
def main():
    st.title("📈 Advanced Momentum Sector Rotation Strategy")
    st.markdown("""
    This dashboard backtests a sector rotation strategy with advanced filters. 
    Use the sidebar to configure the strategy and run the backtest.
    """)

    # --- Sidebar for User Inputs ---
    with st.sidebar:
        st.header("⚙️ Strategy Parameters")

        uploaded_file = st.file_uploader("Upload your Excel data file", type=["xlsx", "xls"])

        if uploaded_file:
            @st.cache_data
            def load_data(file):
                df = pd.read_excel(file, parse_dates=['Date'])
                return df
            
            sample_df = load_data(uploaded_file)
            
            st.subheader("General Settings")
            min_date, max_date = sample_df['Date'].min().date(), sample_df['Date'].max().date()
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
            
            potential_cols = [col for col in sample_df.columns if col != 'Date']
            benchmark_col = st.selectbox("Select Benchmark Column", potential_cols, index=len(potential_cols)-1)
            all_sectors = [col for col in sample_df.columns if col not in ['Date', benchmark_col]]
            
            num_sectors_to_invest = st.slider("Number of Top Sectors to Select", 1, len(all_sectors), min(2, len(all_sectors)))
            
            # --- NEW: Strategy Improvement Filters ---
            st.subheader("Strategy Filters (for Aggressiveness/Safety)")
            use_absolute_momentum = st.checkbox("Use Absolute Momentum Filter", value=True)
            abs_mom_lookback = st.number_input("Absolute Momentum Lookback (days)", value=63, help="Only invest in sectors with positive returns over this period.")
            
            use_regime_filter = st.checkbox("Use Market Regime Filter", value=True)
            regime_sma_lookback = st.number_input("Benchmark SMA for Regime Filter", value=200, help="Only invest if benchmark is above this SMA.")

            with st.expander("Indicator Lookback Periods"):
                mom_1m_lookback = st.number_input("1M Momentum (days)", value=21)
                mom_3m_lookback = st.number_input("3M Momentum (days)", value=63)
                mom_6m_lookback = st.number_input("6M Momentum (days)", value=126)
                sma_lookback = st.number_input("Price/SMA Ratio (days)", value=50)
                rsi_lookback = st.number_input("RSI (days)", value=14)

            with st.expander("Indicator Ranking Weights"):
                st.markdown("Define the importance of each factor. They will be normalized.")
                weight_mom_1m = st.slider("1-Month Momentum Weight", 0.0, 1.0, 0.40)
                weight_mom_3m = st.slider("3-Month Momentum Weight", 0.0, 1.0, 0.30)
                weight_mom_6m = st.slider("6-Month Momentum Weight", 0.0, 1.0, 0.10)
                weight_sma_ratio = st.slider("Price/SMA Ratio Weight", 0.0, 1.0, 0.10)
                weight_rsi = st.slider("RSI Weight", 0.0, 1.0, 0.0)
                weight_macd = st.slider("MACD Weight", 0.0, 1.0, 0.0)
                # --- NEW: Dynamic Volatility Factor ---
                volatility_factor = st.slider("Volatility Factor", -1.0, 1.0, -0.1, help="-1=Prefer Low Vol, 0=Ignore, 1=Prefer High Vol")

            # --- NEW: Total Weights Check ---
            total_weight = weight_mom_1m + weight_mom_3m + weight_mom_6m + weight_sma_ratio + weight_rsi + weight_macd + abs(volatility_factor)
            st.info(f"Sum of absolute weights: {total_weight:.2f}")
            if not np.isclose(total_weight, 1.0):
                st.warning("For standard ranking, weights should sum to 1.0.")

        else:
            st.info("Awaiting for an Excel file to be uploaded.")
            return

    if st.sidebar.button("🚀 Run Backtest"):
        # --- 1. Data Preparation ---
        df = sample_df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.loc[str(start_date):str(end_date)]
        df.ffill(inplace=True) 

        sectors = all_sectors
        
        with st.spinner('Running backtest... This might take a moment.'):
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

                # --- NEW: Market Regime Filter Logic ---
                invest_this_month = True
                if use_regime_filter:
                    benchmark_series = hist_data[benchmark_col].dropna()
                    if len(benchmark_series) > regime_sma_lookback:
                        benchmark_sma = benchmark_series.rolling(window=regime_sma_lookback).mean().iloc[-1]
                        current_benchmark_price = benchmark_series.iloc[-1]
                        if current_benchmark_price < benchmark_sma:
                            invest_this_month = False # Market is in downtrend, stay in cash
                    else:
                        invest_this_month = False # Not enough data, stay safe

                if not invest_this_month:
                    historical_selections[start_period.strftime('%Y-%m')] = ["CASH (Regime Filter)"]
                    # Add a series of zeros for this month's returns
                    period_index = df.loc[start_period:end_period].index
                    portfolio_returns.append(pd.Series(0, index=period_index[1:]))
                    continue

                indicator_values = {}
                for sector in sectors:
                    series = hist_data[sector].dropna()
                    if series.empty: continue
                    
                    sma_val = calculate_sma(series, sma_lookback)
                    sma_ratio = series.iloc[-1] / sma_val if sma_val and not np.isnan(sma_val) and sma_val != 0 else np.nan
                    
                    indicator_values[sector] = {
                        'mom_1m': calculate_momentum(series, mom_1m_lookback),
                        'mom_3m': calculate_momentum(series, mom_3m_lookback),
                        'mom_6m': calculate_momentum(series, mom_6m_lookback),
                        'sma_ratio': sma_ratio,
                        'rsi': calculate_rsi(series, rsi_lookback),
                        'macd': calculate_macd(series)[0],
                        'volatility': calculate_volatility(series, 21),
                        'abs_mom': calculate_momentum(series, abs_mom_lookback) # For filtering
                    }

                indicator_df = pd.DataFrame(indicator_values).T.dropna()
                if indicator_df.empty: continue
                
                # --- NEW: Absolute Momentum Filter ---
                if use_absolute_momentum:
                    indicator_df = indicator_df[indicator_df['abs_mom'] > 0]
                
                if indicator_df.empty or len(indicator_df) < num_sectors_to_invest:
                    historical_selections[start_period.strftime('%Y-%m')] = ["CASH (No Qualifiers)"]
                    period_index = df.loc[start_period:end_period].index
                    portfolio_returns.append(pd.Series(0, index=period_index[1:]))
                    continue

                ranks = pd.DataFrame(index=indicator_df.index)
                ranks['rank_mom_1m'] = indicator_df['mom_1m'].rank(ascending=False)
                ranks['rank_mom_3m'] = indicator_df['mom_3m'].rank(ascending=False)
                ranks['rank_mom_6m'] = indicator_df['mom_6m'].rank(ascending=False)
                ranks['rank_sma_ratio'] = indicator_df['sma_ratio'].rank(ascending=False)
                ranks['rank_rsi'] = indicator_df['rsi'].rank(ascending=False)
                ranks['rank_macd'] = indicator_df['macd'].rank(ascending=False)
                # Volatility rank depends on the factor's sign
                ranks['rank_volatility'] = indicator_df['volatility'].rank(ascending=volatility_factor < 0)

                # Normalize weights to sum to 1
                norm_factor = weight_mom_1m + weight_mom_3m + weight_mom_6m + weight_sma_ratio + weight_rsi + weight_macd + abs(volatility_factor)
                if norm_factor == 0: norm_factor = 1 # Avoid division by zero
                
                ranks['composite_score'] = (
                    (weight_mom_1m / norm_factor) * ranks['rank_mom_1m'] +
                    (weight_mom_3m / norm_factor) * ranks['rank_mom_3m'] +
                    (weight_mom_6m / norm_factor) * ranks['rank_mom_6m'] +
                    (weight_sma_ratio / norm_factor) * ranks['rank_sma_ratio'] +
                    (weight_rsi / norm_factor) * ranks['rank_rsi'] +
                    (weight_macd / norm_factor) * ranks['rank_macd'] +
                    (abs(volatility_factor) / norm_factor) * ranks['rank_volatility']
                )
                
                top_sectors = ranks.sort_values('composite_score').head(num_sectors_to_invest).index.tolist()
                
                historical_selections[start_period.strftime('%Y-%m')] = top_sectors
                
                investment_period_df = df.loc[start_period:end_period]
                daily_returns_selected_sectors = investment_period_df[top_sectors].pct_change().dropna()
                monthly_portfolio_returns = daily_returns_selected_sectors.mean(axis=1)
                portfolio_returns.append(monthly_portfolio_returns)
            
            if not portfolio_returns:
                st.error("Backtest generated no returns. Check date range and parameters.")
                return

            # --- 6. Consolidate and Analyze Results ---
            strategy_returns = pd.concat(portfolio_returns).sort_index()
            strategy_returns = strategy_returns.loc[~strategy_returns.index.duplicated(keep='first')] # Remove duplicates
            strategy_cumulative = (1 + strategy_returns).cumprod()
            
            benchmark_returns = df[benchmark_col].pct_change().loc[strategy_returns.index]
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
                if last_month_selections and "CASH" not in ' '.join(last_month_selections):
                    turnover = len(set(selections) - set(last_month_selections))
                    total_turnover += turnover
                last_month_selections = selections
            num_trading_months = sum(1 for s in historical_selections.values() if "CASH" not in ' '.join(s))
            churn_ratio = total_turnover / (num_trading_months * num_sectors_to_invest) if num_trading_months > 0 else 0

            # --- 7. Display Results ---
            st.header("📊 Backtest Results")
            # Display metrics, equity curve, etc. as before...

            # --- NEW: Current Momentum Scores Table ---
            st.subheader("Current Market Snapshot")
            st.markdown(f"Indicator values and ranks as of the last data point ({df.index[-1].strftime('%Y-%m-%d')}).")
            
            current_indicators = {}
            for sector in sectors:
                series = df[sector].dropna()
                if series.empty: continue
                sma_val = calculate_sma(series, sma_lookback)
                sma_ratio = series.iloc[-1] / sma_val if sma_val and not np.isnan(sma_val) and sma_val != 0 else np.nan
                current_indicators[sector] = {
                    "Price": series.iloc[-1],
                    "1M Mom": calculate_momentum(series, mom_1m_lookback),
                    "3M Mom": calculate_momentum(series, mom_3m_lookback),
                    "6M Mom": calculate_momentum(series, mom_6m_lookback),
                    "RSI": calculate_rsi(series, rsi_lookback),
                    "Volatility": calculate_volatility(series, 21),
                }

            current_df = pd.DataFrame(current_indicators).T.dropna()
            # Add ranks
            current_df['1M_Rank'] = current_df['1M Mom'].rank(ascending=False)
            current_df['3M_Rank'] = current_df['3M Mom'].rank(ascending=False)
            current_df['6M_Rank'] = current_df['6M Mom'].rank(ascending=False)
            # Style and display
            st.dataframe(current_df.style.format(
                {'Price': '{:,.2f}', '1M Mom': '{:.2%}', '3M Mom': '{:.2%}', '6M Mom': '{:.2%}', 'RSI': '{:.1f}', 'Volatility': '{:.2%}'}
            ).background_gradient(cmap='viridis', subset=['1M_Rank', '3M_Rank', '6M_Rank']))


            # --- NEW: Monthly Returns with Benchmark ---
            with st.expander("View Monthly Returns Breakdown (Strategy vs. Benchmark)"):
                strat_monthly = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1).to_frame('Strategy')
                bench_monthly = benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1).to_frame('Benchmark')
                monthly_compare_df = pd.concat([strat_monthly, bench_monthly], axis=1).dropna()
                monthly_compare_df.index = monthly_compare_df.index.strftime('%Y-%m')
                st.dataframe(monthly_compare_df.style.format("{:.2%}"))

# Make sure all helper functions are included above this
if __name__ == '__main__':
    main()