import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Momentum Strategy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- All calculation functions ---
def calculate_momentum(series, n_days):
    """Calculates n-day momentum."""
    if len(series) < n_days + 1:
        return np.nan
    return (series.iloc[-1] / series.iloc[-n_days - 1]) - 1

def calculate_sma(series, window):
    """Calculates the entire Simple Moving Average series."""
    if len(series) < window:
        # Return an empty series of the same type to avoid errors
        return pd.Series(dtype='float64')
    return series.rolling(window=window).mean()

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    """Calculates MACD and Signal Line values."""
    if len(series) < slow_period:
        return np.nan, np.nan
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line.iloc[-1], signal_line.iloc[-1]

def calculate_cagr(series):
    """Calculates Compound Annual Growth Rate."""
    if series.empty or series.iloc[0] == 0:
        return 0
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    num_years = (series.index[-1] - series.index[0]).days / 365.25
    return (end_val / start_val) ** (1 / num_years) - 1 if num_years > 0 else 0

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculates Sharpe Ratio."""
    if returns.empty:
        return 0.0
    annualized_std = returns.std() * np.sqrt(252)
    if annualized_std == 0:
        return np.nan
    annualized_return = returns.mean() * 252
    return (annualized_return - risk_free_rate) / annualized_std

def calculate_max_drawdown(series):
    """Calculates Maximum Drawdown from a cumulative series."""
    if series.empty:
        return 0.0
    return (series / series.cummax() - 1).min()

def calculate_drawdown_series(series):
    """Calculates the full drawdown series for charting."""
    if series.empty:
        return pd.Series(dtype=float)
    cumulative_max = series.cummax()
    drawdown = (series - cumulative_max) / cumulative_max
    return drawdown

# --- Main Application ---
def main():
    st.title("🚀 Intelligent Momentum Sector Strategy")
    st.markdown("This dashboard uses advanced filters and ranking to build and backtest a sector rotation strategy.")

    with st.sidebar:
        st.header("⚙️ Strategy Parameters")
        uploaded_file = st.file_uploader("Upload your Excel data file", type=["xlsx", "xls"])
        
        if not uploaded_file:
            st.info("Please upload an Excel file to begin.")
            return

        @st.cache_data
        def load_data(file):
            return pd.read_excel(file, parse_dates=['Date'])
        
        sample_df = load_data(uploaded_file)
        
        st.subheader("General Settings")
        min_date = sample_df['Date'].min().date()
        max_date = sample_df['Date'].max().date()
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        all_cols = [col for col in sample_df.columns if col != 'Date']
        benchmark_col = st.selectbox("Select Benchmark", all_cols, index=len(all_cols) - 1 if all_cols else 0)
        all_sectors = [col for col in all_cols if col != benchmark_col]
        
        risk_off_options = ["CASH (0% Return)"] + all_cols
        risk_off_asset = st.selectbox(
            "Select 'Risk-Off' Asset", 
            risk_off_options, 
            0, 
            help="Asset to hold when Market Regime Filter is active."
        )
        
        st.subheader("Strategy Configuration")
        num_sectors_to_invest = st.slider(
            "Number of Top Sectors to Select", 
            1, 
            len(all_sectors) if all_sectors else 1, 
            min(2, len(all_sectors) if all_sectors else 1)
        )
        
        concentration_bonus = st.slider(
            "Top Pick Concentration Bonus %", 
            0, 50, 10, 5, 
            help="Overweight the #1 ranked sector by this percentage."
        )

        st.subheader("Filters & Triggers")
        use_regime_filter = st.checkbox("Use Market Regime Filter", value=True)
        regime_sma_lookback = st.number_input("Benchmark SMA for Regime Filter", value=200, min_value=10)
        
        st.subheader("Ranking Logic")
        sma_strategy = st.radio(
            "Select SMA Strategy",
            ["Price / SMA Ratio", "SMA Crossover", "SMA Rate of Change (ROC)"],
            index=1, 
            help="**Crossover**: Ranks Fast SMA / Slow SMA. **ROC**: Ranks the slope of the SMA."
        )
        with st.expander("Indicator Lookbacks & Weights"):
            mom_1m_lookback = st.number_input("1M Mom Lookback", 21)
            mom_3m_lookback = st.number_input("3M Mom Lookback", 63)
            fast_sma_lookback = st.number_input("Fast SMA Lookback", 50)
            slow_sma_lookback = st.number_input("Slow SMA Lookback", 200)
            st.markdown("---")
            weight_mom_1m = st.slider("1M Momentum Wt", 0.0, 1.0, 0.4)
            weight_mom_3m = st.slider("3M Momentum Wt", 0.0, 1.0, 0.4)
            weight_sma_strategy = st.slider("SMA Strategy Wt", 0.0, 1.0, 0.2)
            weight_macd = st.slider("MACD Weight", 0.0, 1.0, 0.0, help="Ranks based on MACD line value.")
            
    if st.sidebar.button("🚀 Run Backtest"):
        df = sample_df.set_index(pd.to_datetime(sample_df['Date'])).drop('Date', axis=1, errors='ignore')
        df = df.loc[str(start_date):str(end_date)].ffill()

        with st.spinner('Running backtest... This is the exciting part!'):
            rebalance_dates = df.resample('MS').first().index
            portfolio_returns = []
            historical_selections = {}

            for i in range(len(rebalance_dates) - 1):
                ranking_date, start_period, end_period = rebalance_dates[i], rebalance_dates[i], rebalance_dates[i+1]
                hist_data = df.loc[:ranking_date]

                invest_this_month = True
                if use_regime_filter:
                    bench_series = hist_data[benchmark_col].dropna()
                    if len(bench_series) > regime_sma_lookback:
                        if bench_series.iloc[-1] < bench_series.rolling(regime_sma_lookback).mean().iloc[-1]:
                            invest_this_month = False
                    else: 
                        invest_this_month = False

                if not invest_this_month:
                    selection_text = f"RISK-OFF ({risk_off_asset})"
                    historical_selections[start_period.strftime('%Y-%m')] = [selection_text]
                    period_index = df.loc[start_period:end_period].index[1:]
                    if risk_off_asset == "CASH (0% Return)":
                        portfolio_returns.append(pd.Series(0, index=period_index))
                    else:
                        risk_off_returns = df[risk_off_asset].loc[start_period:end_period].pct_change().dropna()
                        portfolio_returns.append(risk_off_returns)
                    continue

                indicator_values = {}
                for sector in all_sectors:
                    series = hist_data[sector].dropna()
                    if len(series) < slow_sma_lookback + 5: continue # Ensure enough data
                    
                    sma_fast = calculate_sma(series, fast_sma_lookback)
                    sma_slow = calculate_sma(series, slow_sma_lookback)
                    sma_metric = np.nan
                    
                    if not sma_fast.empty:
                        if sma_strategy == "Price / SMA Ratio" and sma_fast.iloc[-1] != 0:
                            sma_metric = series.iloc[-1] / sma_fast.iloc[-1]
                        elif sma_strategy == "SMA Crossover" and not sma_slow.empty and sma_slow.iloc[-1] != 0:
                            sma_metric = sma_fast.iloc[-1] / sma_slow.iloc[-1]
                        elif sma_strategy == "SMA Rate of Change (ROC)":
                            roc_period = 21 # ~1 month
                            if len(sma_fast) > roc_period:
                                sma_metric = (sma_fast.iloc[-1] / sma_fast.iloc[-roc_period -1]) - 1

                    indicator_values[sector] = {
                        'mom_1m': calculate_momentum(series, mom_1m_lookback),
                        'mom_3m': calculate_momentum(series, mom_3m_lookback),
                        'sma_metric': sma_metric,
                        'macd': calculate_macd(series)[0]
                    }

                indicator_df = pd.DataFrame(indicator_values).T.dropna()
                if indicator_df.empty or len(indicator_df) < num_sectors_to_invest:
                    historical_selections[start_period.strftime('%Y-%m')] = ["CASH (No Qualifiers)"]
                    portfolio_returns.append(pd.Series(0, index=df.loc[start_period:end_period].index[1:]))
                    continue

                ranks = pd.DataFrame(index=indicator_df.index)
                ranks['rank_mom_1m'] = indicator_df['mom_1m'].rank(ascending=False)
                ranks['rank_mom_3m'] = indicator_df['mom_3m'].rank(ascending=False)
                ranks['rank_sma'] = indicator_df['sma_metric'].rank(ascending=False)
                ranks['rank_macd'] = indicator_df['macd'].rank(ascending=False)
                
                total_weight = weight_mom_1m + weight_mom_3m + weight_sma_strategy + weight_macd
                norm_factor = total_weight if total_weight > 0 else 1
                ranks['composite_score'] = ( (weight_mom_1m / norm_factor) * ranks['rank_mom_1m'] +
                                             (weight_mom_3m / norm_factor) * ranks['rank_mom_3m'] +
                                             (weight_sma_strategy / norm_factor) * ranks['rank_sma'] +
                                             (weight_macd / norm_factor) * ranks['rank_macd'] )
                
                top_sectors = ranks.sort_values('composite_score').head(num_sectors_to_invest).index.tolist()
                historical_selections[start_period.strftime('%Y-%m')] = top_sectors
                
                weights = pd.Series(1.0 / num_sectors_to_invest, index=top_sectors)
                if concentration_bonus > 0 and len(weights) > 1:
                    bonus = concentration_bonus / 100.0
                    top_pick = top_sectors[0]
                    reduction_per_sector = bonus / (len(weights) - 1)
                    weights[weights.index != top_pick] -= reduction_per_sector
                    weights[top_pick] += bonus
                
                investment_period_df = df.loc[start_period:end_period, top_sectors]
                monthly_portfolio_returns = (investment_period_df.pct_change().dropna() * weights).sum(axis=1)
                portfolio_returns.append(monthly_portfolio_returns)

            if not portfolio_returns:
                st.error("Backtest generated no returns. Check date range or filter settings."); return

            strategy_returns = pd.concat(portfolio_returns).sort_index().loc[lambda x: ~x.index.duplicated(keep='first')]
            benchmark_returns = df[benchmark_col].pct_change().loc[strategy_returns.index]
            
            strategy_cumulative = (1 + strategy_returns).cumprod()
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            strategy_cagr = calculate_cagr(strategy_cumulative)
            benchmark_cagr = calculate_cagr(benchmark_cumulative)
            strategy_sharpe = calculate_sharpe_ratio(strategy_returns)
            benchmark_sharpe = calculate_sharpe_ratio(benchmark_returns)
            strategy_mdd = calculate_max_drawdown(strategy_cumulative)
            benchmark_mdd = calculate_max_drawdown(benchmark_cumulative)

            st.header("📊 Backtest Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Strategy CAGR", f"{strategy_cagr:.2%}")
            col1.metric(f"{benchmark_col} CAGR", f"{benchmark_cagr:.2%}")
            col2.metric("Strategy Sharpe Ratio", f"{strategy_sharpe:.2f}")
            col2.metric(f"{benchmark_col} Sharpe Ratio", f"{benchmark_sharpe:.2f}")
            col3.metric("Strategy Max Drawdown", f"{strategy_mdd:.2%}")
            col3.metric(f"{benchmark_col} Max Drawdown", f"{benchmark_mdd:.2%}")
            
            st.subheader("Growth of Initial Investment")
            
            # --- THE FIX IS HERE ---
            initial_investment = st.number_input(
                "Initial Investment Amount", 
                min_value=1000, 
                value=10000, 
                step=1000, 
                key="growth_investment"
            )
            # --- END OF FIX ---

            fig_growth = go.Figure()
            fig_growth.add_trace(go.Scatter(x=strategy_cumulative.index, y=strategy_cumulative * initial_investment, name='Strategy', line=dict(color='royalblue', width=2)))
            fig_growth.add_trace(go.Scatter(x=benchmark_cumulative.index, y=benchmark_cumulative * initial_investment, name=benchmark_col, line=dict(color='grey', dash='dash')))
            fig_growth.update_layout(title_text=f'Growth of ₹{initial_investment:,.0f}', yaxis_title='Portfolio Value (₹)')
            st.plotly_chart(fig_growth, use_container_width=True)

            st.subheader("Drawdown Periods")
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=strategy_cumulative.index, y=calculate_drawdown_series(strategy_cumulative), fill='tozeroy', mode='lines', name='Strategy Drawdown', line=dict(color='crimson')))
            fig_dd.add_trace(go.Scatter(x=benchmark_cumulative.index, y=calculate_drawdown_series(benchmark_cumulative), mode='lines', name='Benchmark Drawdown', line=dict(color='grey', dash='dash')))
            fig_dd.update_layout(title_text='Strategy vs. Benchmark Drawdowns', yaxis_title='Drawdown', yaxis_tickformat='.0%')
            st.plotly_chart(fig_dd, use_container_width=True)

            st.header("📋 Detailed Data & History")
            tab1, tab2 = st.tabs(["Historical Selections", "Monthly Returns"])
            with tab1:
                st.write("Sectors selected at the beginning of each month.")
                selections_df = pd.DataFrame.from_dict(historical_selections, orient='index')
                selections_df.index.name = 'Month'
                st.dataframe(selections_df, use_container_width=True)
            with tab2:
                st.write("Monthly performance comparison.")
                strat_monthly = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1).to_frame('Strategy')
                bench_monthly = benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1).to_frame('Benchmark')
                monthly_df = pd.concat([strat_monthly, bench_monthly], axis=1).dropna()
                monthly_df['Outperformance'] = monthly_df['Strategy'] - monthly_df['Benchmark']
                monthly_df.index = monthly_df.index.strftime('%Y-%m')
                st.dataframe(monthly_df.style.format('{:.2%}').background_gradient(cmap='RdYlGn', subset=['Outperformance']), use_container_width=True)
                
if __name__ == '__main__':
    main()