import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Comprehensive Momentum Sector Strategy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Indicator Calculation Functions ---
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

# --- Performance Metrics ---
def calculate_cagr(series):
    if series.empty or series.iloc[0] == 0: return 0
    start_val = series.iloc[0]; end_val = series.iloc[-1]
    num_years = (series.index[-1] - series.index[0]).days / 365.25
    if num_years == 0: return 0
    return (end_val / start_val) ** (1 / num_years) - 1

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    if returns.empty: return 0.0
    annualized_return = returns.mean() * 252
    annualized_std = returns.std() * np.sqrt(252)
    if annualized_std == 0: return np.nan
    return (annualized_return - risk_free_rate) / annualized_std
    
def calculate_max_drawdown(series):
    if series.empty: return 0.0
    return (series / series.cummax() - 1).min()

# --- NEW: Function to calculate the full drawdown series for charting ---
def calculate_drawdown_series(series):
    """Calculates the drawdown series from a cumulative return series."""
    if series.empty: return pd.Series(dtype=float)
    cumulative_max = series.cummax()
    drawdown = (series - cumulative_max) / cumulative_max
    return drawdown

# --- Main Application ---
def main():
    st.title("📈 Comprehensive Momentum Sector Strategy")
    st.markdown("Use the sidebar to configure parameters, filters, and run the backtest.")

    with st.sidebar:
        # (Sidebar code remains identical to the previous version)
        st.header("⚙️ Strategy Parameters")
        uploaded_file = st.file_uploader("Upload your Excel data file", type=["xlsx", "xls"])
        if not uploaded_file:
            st.info("Awaiting for an Excel file to be uploaded.")
            return

        @st.cache_data
        def load_data(file):
            return pd.read_excel(file, parse_dates=['Date'])
        
        sample_df = load_data(uploaded_file)
        
        st.subheader("General Settings")
        min_date, max_date = sample_df['Date'].min().date(), sample_df['Date'].max().date()
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        potential_cols = [col for col in sample_df.columns if col != 'Date']
        benchmark_col = st.selectbox("Select Benchmark Column", potential_cols, index=len(potential_cols)-1)
        all_sectors = [col for col in sample_df.columns if col not in ['Date', benchmark_col]]
        
        num_sectors_to_invest = st.slider("Number of Top Sectors to Select", 1, len(all_sectors), min(2, len(all_sectors)))
        
        st.subheader("Strategy Filters")
        use_absolute_momentum = st.checkbox("Use Absolute Momentum Filter", value=True)
        abs_mom_lookback = st.number_input("Absolute Momentum Lookback (days)", value=63)
        
        use_regime_filter = st.checkbox("Use Market Regime Filter", value=True)
        regime_sma_lookback = st.number_input("Benchmark SMA for Regime Filter", value=200)

        with st.expander("Indicator Lookback Periods"):
            mom_1m_lookback, mom_3m_lookback, mom_6m_lookback = st.number_input("1M Mom (d)", 21), st.number_input("3M Mom (d)", 63), st.number_input("6M Mom (d)", 126)
            sma_lookback, rsi_lookback = st.number_input("SMA (d)", 50), st.number_input("RSI (d)", 14)

        with st.expander("Indicator Ranking Weights"):
            weight_mom_1m = st.slider("1M Mom Wt", 0.0, 1.0, 0.40)
            weight_mom_3m = st.slider("3M Mom Wt", 0.0, 1.0, 0.30)
            weight_mom_6m = st.slider("6M Mom Wt", 0.0, 1.0, 0.10)
            weight_sma_ratio = st.slider("Price/SMA Wt", 0.0, 1.0, 0.10)
            volatility_factor = st.slider("Volatility Factor", -1.0, 1.0, -0.1, help="-1=Low Vol, 0=Ignore, 1=High Vol")
        
        total_weight = weight_mom_1m + weight_mom_3m + weight_mom_6m + weight_sma_ratio + abs(volatility_factor)
        st.info(f"Sum of absolute weights: {total_weight:.2f}")
        if not np.isclose(total_weight, 1.0) and total_weight > 0:
            st.warning("Weights will be normalized to sum to 1.0 for ranking.")
            
    if st.sidebar.button("🚀 Run Backtest"):
        # --- 1. Data Prep & Backtest Loop (Same as before) ---
        df = sample_df.copy()
        df = df.set_index(pd.to_datetime(df['Date'])).drop('Date', axis=1)
        df = df.loc[str(start_date):str(end_date)]
        df.ffill(inplace=True) 

        # ... (The entire backtesting loop is the same as the previous version)
        # ... it calculates `strategy_returns`, `benchmark_returns`, etc.
        # --- [For brevity, the ~100 lines of the backtest loop are omitted here, but they are essential and should be copied from the previous answer] ---
        # For a complete working script, this loop must be included. I'll just show the calculation part
        # starting from the end of the loop to the display.
        
        # --- This block is a summary of the backtest loop needed here ---
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
            invest_this_month = True
            if use_regime_filter:
                benchmark_series = hist_data[benchmark_col].dropna()
                if len(benchmark_series) > regime_sma_lookback:
                    if benchmark_series.iloc[-1] < benchmark_series.rolling(regime_sma_lookback).mean().iloc[-1]:
                        invest_this_month = False
                else: invest_this_month = False
            if not invest_this_month:
                historical_selections[start_period.strftime('%Y-%m')] = ["CASH (Regime)"]
                period_index = df.loc[start_period:end_period].index
                portfolio_returns.append(pd.Series(0, index=period_index[1:]))
                continue
            indicator_values = {}
            for sector in all_sectors:
                series = hist_data[sector].dropna()
                if series.empty: continue
                indicator_values[sector] = {
                    'mom_1m': calculate_momentum(series, mom_1m_lookback), 'mom_3m': calculate_momentum(series, mom_3m_lookback),
                    'mom_6m': calculate_momentum(series, mom_6m_lookback), 'sma_ratio': series.iloc[-1] / calculate_sma(series, sma_lookback) if calculate_sma(series, sma_lookback) else np.nan,
                    'volatility': calculate_volatility(series, 21), 'abs_mom': calculate_momentum(series, abs_mom_lookback)
                }
            indicator_df = pd.DataFrame(indicator_values).T.dropna()
            if use_absolute_momentum: indicator_df = indicator_df[indicator_df['abs_mom'] > 0]
            if indicator_df.empty or len(indicator_df) < num_sectors_to_invest:
                historical_selections[start_period.strftime('%Y-%m')] = ["CASH (Filters)"]
                period_index = df.loc[start_period:end_period].index
                portfolio_returns.append(pd.Series(0, index=period_index[1:]))
                continue
            ranks = pd.DataFrame(index=indicator_df.index)
            ranks['rank_mom_1m'] = indicator_df['mom_1m'].rank(ascending=False)
            ranks['rank_mom_3m'] = indicator_df['mom_3m'].rank(ascending=False)
            ranks['rank_mom_6m'] = indicator_df['mom_6m'].rank(ascending=False)
            ranks['rank_sma_ratio'] = indicator_df['sma_ratio'].rank(ascending=False)
            ranks['rank_volatility'] = indicator_df['volatility'].rank(ascending=volatility_factor < 0)
            norm_factor = weight_mom_1m + weight_mom_3m + weight_mom_6m + weight_sma_ratio + abs(volatility_factor)
            if norm_factor == 0: norm_factor = 1
            ranks['composite_score'] = (
                (weight_mom_1m / norm_factor) * ranks['rank_mom_1m'] + (weight_mom_3m / norm_factor) * ranks['rank_mom_3m'] +
                (weight_mom_6m / norm_factor) * ranks['rank_mom_6m'] + (weight_sma_ratio / norm_factor) * ranks['rank_sma_ratio'] +
                (abs(volatility_factor) / norm_factor) * ranks['rank_volatility'] )
            top_sectors = ranks.sort_values('composite_score').head(num_sectors_to_invest).index.tolist()
            historical_selections[start_period.strftime('%Y-%m')] = top_sectors
            investment_period_df = df.loc[start_period:end_period]
            monthly_portfolio_returns = investment_period_df[top_sectors].pct_change().dropna().mean(axis=1)
            portfolio_returns.append(monthly_portfolio_returns)
        # --- END OF BACKTEST LOOP ---

        if not portfolio_returns:
            st.error("Backtest generated no returns. Check date range and filters.")
            return

        # --- 6. Analysis and Display ---
        strategy_returns = pd.concat(portfolio_returns).sort_index()
        strategy_returns = strategy_returns.loc[~strategy_returns.index.duplicated(keep='first')]
        
        benchmark_returns = df[benchmark_col].pct_change().loc[strategy_returns.index]
        
        # --- Calculate all metrics and series for charts ---
        strategy_cumulative = (1 + strategy_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        strategy_cagr = calculate_cagr(strategy_cumulative)
        benchmark_cagr = calculate_cagr(benchmark_cumulative)
        strategy_sharpe = calculate_sharpe_ratio(strategy_returns)
        benchmark_sharpe = calculate_sharpe_ratio(benchmark_returns)
        strategy_mdd = calculate_max_drawdown(strategy_cumulative)
        benchmark_mdd = calculate_max_drawdown(benchmark_cumulative)
        strategy_drawdown_series = calculate_drawdown_series(strategy_cumulative)
        benchmark_drawdown_series = calculate_drawdown_series(benchmark_cumulative)

        # Churn Ratio Calculation
        total_turnover = sum(len(set(selections) - set(last_month_selections)) for month, selections in historical_selections.items() if (last_month_selections := historical_selections.get(pd.to_datetime(month).to_period('M').start_time - pd.DateOffset(months=1), [])) and "CASH" not in ' '.join(selections))
        num_trading_months = sum(1 for s in historical_selections.values() if "CASH" not in ' '.join(s))
        churn_ratio = total_turnover / (num_trading_months * num_sectors_to_invest) if num_trading_months > 0 else 0

        # --- 7. Display Results ---
        st.header("📊 Backtest Results")

        # --- RESTORED: Key Performance Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strategy CAGR", f"{strategy_cagr:.2%}")
        col1.metric(f"{benchmark_col} CAGR", f"{benchmark_cagr:.2%}")
        col2.metric("Strategy Sharpe", f"{strategy_sharpe:.2f}")
        col2.metric(f"{benchmark_col} Sharpe", f"{benchmark_sharpe:.2f}")
        col3.metric("Strategy Max Drawdown", f"{strategy_mdd:.2%}")
        col3.metric(f"{benchmark_col} Max Drawdown", f"{benchmark_mdd:.2%}")
        col4.metric("Strategy Churn %", f"{churn_ratio:.2%}", help="Avg % of portfolio replaced monthly.")
        col4.metric("Annualized Volatility", f"{(strategy_returns.std() * np.sqrt(252)):.2%}")

        # --- NEW: Growth of Investment Chart ---
        st.subheader("Growth of Initial Investment")
        initial_investment = st.number_input("Initial Investment Amount", value=10000, step=1000, key="growth_input")
        
        fig_growth = go.Figure()
        fig_growth.add_trace(go.Scatter(x=strategy_cumulative.index, y=strategy_cumulative * initial_investment, mode='lines', name='Strategy', line=dict(color='blue', width=2)))
        fig_growth.add_trace(go.Scatter(x=benchmark_cumulative.index, y=benchmark_cumulative * initial_investment, mode='lines', name=benchmark_col, line=dict(color='grey', dash='dash')))
        fig_growth.update_layout(title_text=f'Growth of ₹{initial_investment:,.0f}', yaxis_title='Portfolio Value (₹)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_growth, use_container_width=True)

        # --- NEW: Drawdown Chart ---
        st.subheader("Drawdown Periods")
        fig_drawdown = go.Figure()
        fig_drawdown.add_trace(go.Scatter(x=strategy_drawdown_series.index, y=strategy_drawdown_series, fill='tozeroy', mode='lines', name='Strategy Drawdown', line=dict(color='red')))
        fig_drawdown.add_trace(go.Scatter(x=benchmark_drawdown_series.index, y=benchmark_drawdown_series, fill='tozeroy', mode='lines', name=f'{benchmark_col} Drawdown', line=dict(color='grey')))
        fig_drawdown.update_layout(title_text='Strategy vs. Benchmark Drawdowns', yaxis_title='Drawdown', yaxis_tickformat='.0%', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_drawdown, use_container_width=True)

        # --- Current Snapshot Table ---
        st.subheader("Current Market Snapshot")
        # (This section is unchanged and should be copied from the previous answer)
        
        # --- Historical Data Expanders ---
        # (This section is also unchanged)
        
if __name__ == '__main__':
    main()