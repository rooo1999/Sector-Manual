# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- Indicator Calculation Functions (Refined Logic) ---

def calculate_momentum(series, n):
    """Calculates n-period price momentum."""
    if len(series) > n:
        return (series.iloc[-1] / series.iloc[-n-1]) - 1 if series.iloc[-n-1] != 0 else 0
    return 0

def calculate_rsi_momentum(series, n=14, roc_period=5):
    """Calculates the momentum of the RSI value itself (its rate of change)."""
    if len(series) < n + roc_period:
        return np.nan
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=n-1, min_periods=n).mean()
    avg_loss = loss.ewm(com=n-1, min_periods=n).mean()
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    if len(rsi_series) > roc_period:
        return rsi_series.iloc[-1] - rsi_series.iloc[-1 - roc_period]
    return np.nan

def calculate_macd_acceleration(series, fast_period=12, slow_period=26, signal_period=9, roc_period=5):
    """Calculates the acceleration of the MACD Histogram (its rate of change)."""
    if len(series) < slow_period + signal_period + roc_period:
        return np.nan
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    if len(macd_histogram) > roc_period:
        return macd_histogram.iloc[-1] - macd_histogram.iloc[-1 - roc_period]
    return np.nan


# --- Backtesting Engine (Final Version with Equal Weighting) ---

def run_backtest(df, benchmark_col, start_date, end_date, lookback_days, top_n):
    df_filtered = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))].copy()
    sector_cols = [col for col in df.columns if col != benchmark_col]
    
    monthly_index_series = df_filtered.index.to_series()
    rebal_dates = monthly_index_series.groupby(pd.Grouper(freq='M')).max().dropna().tolist()

    portfolio_returns = []
    historical_selections = {}

    for i in range(len(rebal_dates) - 1):
        rebal_date = rebal_dates[i]
        next_rebal_date = rebal_dates[i+1]
        
        ranking_data = {}
        for sector in sector_cols:
            series = df.loc[:rebal_date, sector]
            if len(series) < 70: continue

            ranking_data[sector] = {
                "mom_1m": calculate_momentum(series, 21),
                "mom_3m": calculate_momentum(series, 63),
                "rsi_mom": calculate_rsi_momentum(series, n=lookback_days),
                "macd_accel": calculate_macd_acceleration(series),
            }
        
        if not ranking_data: continue
        ranking_df = pd.DataFrame(ranking_data).T.dropna()
        if ranking_df.empty: continue

        # Rank each factor
        rank_cols = []
        for col in ranking_df.columns:
            rank_col_name = f'rank_{col}'
            ranking_df[rank_col_name] = ranking_df[col].rank(ascending=False)
            rank_cols.append(rank_col_name)
        
        # --- MODIFICATION: REMOVED WEIGHTS ---
        # Calculate composite rank by simply summing the individual ranks (equal weighting)
        ranking_df['composite_rank'] = ranking_df[rank_cols].sum(axis=1)
        
        top_sectors = ranking_df.sort_values('composite_rank').head(top_n).index.tolist()
        historical_selections[next_rebal_date.strftime('%Y-%m')] = top_sectors

        period_data = df_filtered.loc[rebal_date:next_rebal_date]
        buy_date_loc = period_data.index.get_loc(rebal_date) + 1
        
        if buy_date_loc < len(period_data.index):
            holding_start_date = period_data.index[buy_date_loc]
            holding_end_date = next_rebal_date
            returns = period_data.loc[holding_end_date, top_sectors] / period_data.loc[holding_start_date, top_sectors] - 1
            monthly_return = returns.mean() if not top_sectors else returns.mean()
        else:
            monthly_return = 0
        
        portfolio_returns.append({'Date': next_rebal_date, 'Strategy': monthly_return})
    
    if not portfolio_returns: return None
        
    returns_df = pd.DataFrame(portfolio_returns).set_index('Date')
    benchmark_monthly = df_filtered.loc[returns_df.index, benchmark_col].pct_change().dropna()
    returns_df['Benchmark'] = benchmark_monthly
    returns_df = returns_df.dropna()
    if returns_df.empty: return None

    strategy_equity = (1 + returns_df['Strategy']).cumprod() * 100
    benchmark_equity = (1 + returns_df['Benchmark']).cumprod() * 100

    num_years = max((returns_df.index[-1] - returns_df.index[0]).days / 365.25, 1)
    
    cagr_strategy = (strategy_equity.iloc[-1] / 100)**(1/num_years) - 1
    vol_strategy = returns_df['Strategy'].std() * np.sqrt(12)
    sharpe_strategy = cagr_strategy / vol_strategy if vol_strategy != 0 else 0
    drawdown_strategy = (strategy_equity / strategy_equity.cummax() - 1).min()
    calmar_strategy = cagr_strategy / abs(drawdown_strategy) if drawdown_strategy != 0 else 0
    
    cagr_benchmark = (benchmark_equity.iloc[-1] / 100)**(1/num_years) - 1
    vol_benchmark = returns_df['Benchmark'].std() * np.sqrt(12)
    sharpe_benchmark = cagr_benchmark / vol_benchmark if vol_benchmark != 0 else 0
    drawdown_benchmark = (benchmark_equity / benchmark_equity.cummax() - 1).min()
    calmar_benchmark = cagr_benchmark / abs(drawdown_benchmark) if drawdown_benchmark != 0 else 0

    turnover = []
    selections_list = list(historical_selections.values())
    for i in range(1, len(selections_list)):
        turnover.append(len(set(selections_list[i]) - set(selections_list[i-1])) / top_n)
    churn_ratio = np.mean(turnover) if turnover else 0

    metrics = {
        'Metric': ['CAGR', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio'],
        'Strategy': [cagr_strategy, vol_strategy, sharpe_strategy, drawdown_strategy, calmar_strategy],
        'Benchmark': [cagr_benchmark, vol_benchmark, sharpe_benchmark, drawdown_benchmark, calmar_benchmark]
    }
    
    return {
        "strategy_equity": strategy_equity, "benchmark_equity": benchmark_equity,
        "metrics": pd.DataFrame(metrics), "monthly_returns": returns_df,
        "historical_selections": pd.DataFrame.from_dict(historical_selections, orient='index', columns=[f'Top_{i+1}' for i in range(top_n)]),
        "churn_ratio": churn_ratio
    }

# --- Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title("Equal-Weight Momentum Sector Rotation Strategy")
st.markdown("This model uses an equal-weight combination of **1M/3M Momentum**, **RSI Momentum**, and **MACD Acceleration** to rank sectors.")

st.sidebar.header("Strategy Controls")
uploaded_file = st.sidebar.file_uploader("Upload your sectoral data (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, engine='openpyxl')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    all_cols = df.columns.tolist()

    st.sidebar.subheader("General Settings")
    start_date = st.sidebar.date_input("Start Date", df.index.min().date())
    end_date = st.sidebar.date_input("End Date", df.index.max().date())
    benchmark_col = st.sidebar.selectbox("Select Benchmark Column", all_cols, index=len(all_cols)-1)
    lookback_days = st.sidebar.number_input("RSI Lookback Period (Days)", min_value=10, max_value=50, value=14, step=1, help="The period for the base RSI calculation.")
    top_n = st.sidebar.number_input("Number of Sectors to Select", min_value=1, max_value=10, value=2, step=1)

    if st.sidebar.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest... This may take a moment."):
            # Pass arguments without the 'weights' dictionary
            results = run_backtest(df, benchmark_col, start_date, end_date, lookback_days, top_n)
        
        if results:
            st.header("Backtest Results")
            
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
            col4.metric("Churn Ratio", f"{results['churn_ratio']:.2%}", help="The average monthly percentage of sectors that are replaced.")

            st.subheader("Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results['strategy_equity'].index, y=results['strategy_equity'], mode='lines', name='Strategy', line=dict(color='royalblue', width=2)))
            fig.add_trace(go.Scatter(x=results['benchmark_equity'].index, y=results['benchmark_equity'], mode='lines', name='Benchmark', line=dict(color='grey', width=2)))
            fig.update_layout(title='Strategy vs. Benchmark: Growth of ₹100', xaxis_title='Date', yaxis_title='Portfolio Value', legend_title_text='Legend')
            st.plotly_chart(fig, use_container_width=True)

            tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Historical Selections", "Monthly Returns"])
            
            with tab1:
                styler = results['metrics'].set_index('Metric').style
                # Apply percentage format to specific rows for all columns
                styler.format(formatter="{:.2%}", subset=(['CAGR', 'Annualized Volatility', 'Max Drawdown'], slice(None)))
                # Apply float format to other specific rows for all columns
                styler.format(formatter="{:.2f}", subset=(['Sharpe Ratio', 'Calmar Ratio'], slice(None)))
                st.dataframe(styler)

            with tab2:
                st.subheader("Monthly Sector Selections")
                st.dataframe(results['historical_selections'])
            
            with tab3:
                st.subheader("Strategy vs. Benchmark Monthly Returns")
                # --- ERROR FIX ---
                # Using the explicit `formatter` keyword prevents the KeyError
                st.dataframe(results['monthly_returns'].style.format(formatter='{:.2%}'))
        else:
            st.error("Backtest failed. Please check your data and parameters. The selected date range may not have enough data for calculations (at least ~4 months is recommended).")
else:
    st.info("Please upload an Excel file with your sectoral data to begin the analysis.")