# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- Indicator Calculation Functions (Revised) ---

def calculate_momentum(series, n):
    """Calculates n-period momentum."""
    if len(series) > n:
        return (series.iloc[-1] / series.iloc[-n-1]) - 1 if series.iloc[-n-1] != 0 else 0
    return 0

def calculate_rsi_momentum(series, n=14, roc_period=5):
    """
    Calculates the momentum of the RSI.
    Improvement: Measures if relative strength is accelerating or decelerating.
    """
    if len(series) < n + roc_period:
        return np.nan
    
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Use EWM for a smoother average
    avg_gain = gain.ewm(com=n-1, min_periods=n).mean()
    avg_loss = loss.ewm(com=n-1, min_periods=n).mean()
    
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    
    # Calculate the rate of change of the RSI
    if len(rsi_series) > roc_period:
        return rsi_series.iloc[-1] - rsi_series.iloc[-1 - roc_period]
    return np.nan

def calculate_macd_acceleration(series, fast_period=12, slow_period=26, signal_period=9, roc_period=5):
    """
    Calculates the acceleration of the MACD Histogram.
    Improvement: Measures if the trend momentum is strengthening or weakening.
    """
    if len(series) < slow_period + signal_period + roc_period:
        return np.nan

    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    # Calculate the rate of change of the MACD Histogram
    if len(macd_histogram) > roc_period:
        return macd_histogram.iloc[-1] - macd_histogram.iloc[-1 - roc_period]
    return np.nan


# --- Backtesting Engine (Revised) ---

def run_backtest(df, benchmark_col, start_date, end_date, lookback_days, top_n, weights):
    df_filtered = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))].copy()
    sector_cols = [col for col in df.columns if col != benchmark_col]
    rebal_dates = df_filtered.resample('M').last().index

    portfolio_returns = []
    historical_selections = {}

    for i in range(len(rebal_dates) - 1):
        rebal_date = rebal_dates[i]
        next_rebal_date = rebal_dates[i+1]
        
        ranking_data = {}
        for sector in sector_cols:
            series = df.loc[:rebal_date, sector]
            
            if len(series) < 70: # Min length for 3M momentum + lookback
                continue

            mom_1m = calculate_momentum(series, 21)
            mom_3m = calculate_momentum(series, 63)
            rsi_mom = calculate_rsi_momentum(series, n=lookback_days)
            macd_accel = calculate_macd_acceleration(series)

            ranking_data[sector] = {
                "mom_1m": mom_1m,
                "mom_3m": mom_3m,
                "rsi_mom": rsi_mom,
                "macd_accel": macd_accel,
            }
        
        if not ranking_data: continue
        ranking_df = pd.DataFrame(ranking_data).T.dropna()
        if ranking_df.empty: continue

        # --- Ranking ---
        ranking_df['rank_mom_1m'] = ranking_df['mom_1m'].rank(ascending=False)
        ranking_df['rank_mom_3m'] = ranking_df['mom_3m'].rank(ascending=False)
        ranking_df['rank_rsi_mom'] = ranking_df['rsi_mom'].rank(ascending=False)
        ranking_df['rank_macd_accel'] = ranking_df['macd_accel'].rank(ascending=False)
        
        # --- Composite Score ---
        ranking_df['composite_rank'] = (
            weights['mom_1m'] * ranking_df['rank_mom_1m'] +
            weights['mom_3m'] * ranking_df['rank_mom_3m'] +
            weights['rsi_mom'] * ranking_df['rank_rsi_mom'] +
            weights['macd_accel'] * ranking_df['rank_macd_accel']
        )
        
        # --- Selection ---
        top_sectors = ranking_df.sort_values('composite_rank').head(top_n).index.tolist()
        historical_selections[next_rebal_date.strftime('%Y-%m')] = top_sectors

        # --- Calculate portfolio return for the next month ---
        period_data = df_filtered.loc[rebal_date:next_rebal_date]
        holding_start_date = period_data.index[1] if len(period_data) > 1 else period_data.index[0]
        holding_end_date = next_rebal_date
        
        if not top_sectors or holding_start_date == holding_end_date:
            monthly_return = 0
        else:
            returns = period_data.loc[holding_end_date, top_sectors] / period_data.loc[holding_start_date, top_sectors] - 1
            monthly_return = returns.mean()
        
        portfolio_returns.append({'Date': next_rebal_date, 'Strategy': monthly_return})
    
    if not portfolio_returns: return None
        
    # --- Performance Analysis ---
    returns_df = pd.DataFrame(portfolio_returns).set_index('Date')
    benchmark_monthly = df_filtered.loc[returns_df.index, benchmark_col].pct_change().dropna()
    returns_df['Benchmark'] = benchmark_monthly
    returns_df = returns_df.dropna()

    strategy_equity = (1 + returns_df['Strategy']).cumprod() * 100
    benchmark_equity = (1 + returns_df['Benchmark']).cumprod() * 100

    num_years = max((returns_df.index[-1] - returns_df.index[0]).days / 365.25, 1)
    
    cagr_strategy = (strategy_equity.iloc[-1] / 100)**(1/num_years) - 1
    vol_strategy = returns_df['Strategy'].std() * np.sqrt(12)
    sharpe_strategy = cagr_strategy / vol_strategy if vol_strategy != 0 else 0
    drawdown_strategy = (strategy_equity / strategy_equity.cummax() - 1).min()
    
    cagr_benchmark = (benchmark_equity.iloc[-1] / 100)**(1/num_years) - 1
    vol_benchmark = returns_df['Benchmark'].std() * np.sqrt(12)
    sharpe_benchmark = cagr_benchmark / vol_benchmark if vol_benchmark != 0 else 0
    drawdown_benchmark = (benchmark_equity / benchmark_equity.cummax() - 1).min()

    turnover = []
    selections_list = list(historical_selections.values())
    for i in range(1, len(selections_list)):
        turnover.append(len(set(selections_list[i]) - set(selections_list[i-1])) / top_n)
    churn_ratio = np.mean(turnover) if turnover else 0

    metrics = {
        'Metric': ['CAGR', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio'],
        'Strategy': [cagr_strategy, vol_strategy, sharpe_strategy, drawdown_strategy, cagr_strategy / abs(drawdown_strategy) if drawdown_strategy != 0 else 0],
        'Benchmark': [cagr_benchmark, vol_benchmark, sharpe_benchmark, drawdown_benchmark, cagr_benchmark / abs(drawdown_benchmark) if drawdown_benchmark != 0 else 0]
    }
    
    results = {
        "strategy_equity": strategy_equity, "benchmark_equity": benchmark_equity,
        "metrics": pd.DataFrame(metrics), "monthly_returns": returns_df,
        "historical_selections": pd.DataFrame.from_dict(historical_selections, orient='index', columns=[f'Top_{i+1}' for i in range(top_n)]),
        "churn_ratio": churn_ratio
    }
    return results

# --- Streamlit UI (Revised) ---

st.set_page_config(layout="wide")
st.title("Enhanced Momentum Sector Rotation Strategy")
st.markdown("This model uses **1M/3M Momentum**, **RSI Momentum**, and **MACD Acceleration** to rank and select top sectors.")

uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, engine='openpyxl')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    all_cols = df.columns.tolist()

    st.sidebar.header("Strategy Parameters")
    st.sidebar.subheader("General Settings")
    
    start_date = st.sidebar.date_input("Start Date", df.index.min().date())
    end_date = st.sidebar.date_input("End Date", df.index.max().date())
    benchmark_col = st.sidebar.selectbox("Select Benchmark", all_cols, index=len(all_cols)-1)
    lookback_days = st.sidebar.number_input("RSI Lookback Period (Days)", min_value=10, max_value=50, value=14, step=1)
    top_n = st.sidebar.number_input("Sectors to Select", min_value=1, max_value=10, value=2, step=1)

    st.sidebar.subheader("Factor Weights")
    weights = {
        'mom_1m': st.sidebar.slider("1-Month Momentum Weight", 0.0, 5.0, 1.0, 0.1),
        'mom_3m': st.sidebar.slider("3-Month Momentum Weight", 0.0, 5.0, 1.0, 0.1),
        'rsi_mom': st.sidebar.slider("RSI Momentum Weight", 0.0, 5.0, 1.0, 0.1),
        'macd_accel': st.sidebar.slider("MACD Acceleration Weight", 0.0, 5.0, 1.0, 0.1)
    }

    if st.sidebar.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            results = run_backtest(df, benchmark_col, start_date, end_date, lookback_days, top_n, weights)
        
        if results:
            st.header("Backtest Results")
            
            # --- Key Metrics ---
            st.subheader("Key Performance Indicators")
            col1, col2, col3, col4 = st.columns(4)
            metrics_data = results['metrics'].set_index('Metric')
            col1.metric("Strategy CAGR", f"{metrics_data.loc['CAGR', 'Strategy']:.2%}")
            col1.metric("Benchmark CAGR", f"{metrics_data.loc['CAGR', 'Benchmark']:.2%}")
            col2.metric("Strategy Sharpe Ratio", f"{metrics_data.loc['Sharpe Ratio', 'Strategy']:.2f}")
            col2.metric("Benchmark Sharpe Ratio", f"{metrics_data.loc['Sharpe Ratio', 'Benchmark']:.2f}")
            col3.metric("Strategy Max Drawdown", f"{metrics_data.loc['Max Drawdown', 'Strategy']:.2%}")
            col3.metric("Benchmark Max Drawdown", f"{metrics_data.loc['Max Drawdown', 'Benchmark']:.2%}")
            col4.metric("Strategy Calmar Ratio", f"{metrics_data.loc['Calmar Ratio', 'Strategy']:.2f}")
            col4.metric("Churn Ratio", f"{results['churn_ratio']:.2%}")

            # --- Equity Curve ---
            st.subheader("Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results['strategy_equity'].index, y=results['strategy_equity'], mode='lines', name='Strategy'))
            fig.add_trace(go.Scatter(x=results['benchmark_equity'].index, y=results['benchmark_equity'], mode='lines', name='Benchmark'))
            fig.update_layout(title='Strategy vs. Benchmark Growth of ₹100', xaxis_title='Date', yaxis_title='Portfolio Value')
            st.plotly_chart(fig, use_container_width=True)

            # --- Detailed Tabs ---
            tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Historical Selections", "Monthly Returns"])
            with tab1:
                st.dataframe(results['metrics'].set_index('Metric').style.format(
                    {'Strategy': '{:.2%}', 'Benchmark': '{:.2%}'}, subset=['CAGR', 'Annualized Volatility', 'Max Drawdown']
                ).format(
                    {'Strategy': '{:.2f}', 'Benchmark': '{:.2f}'}, subset=['Sharpe Ratio', 'Calmar Ratio']
                ))
            with tab2:
                st.subheader("Monthly Sector Selections")
                st.dataframe(results['historical_selections'])
            with tab3:
                st.subheader("Monthly Returns Breakdown")
                st.dataframe(results['monthly_returns'].style.format('{:.2%}'))
        else:
            st.error("Backtest failed. Check if the date range provides enough data for calculations (at least ~4 months).")
else:
    st.info("Upload an Excel file with your sectoral data to start.")