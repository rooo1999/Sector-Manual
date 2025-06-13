import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Momentum Strategy Backtester",
    layout="wide",
)

# --- Calculation Functions ---

@st.cache_data
def load_data(file):
    """Loads and preprocesses data from an Excel file."""
    try:
        df = pd.read_excel(file, index_col='Date', parse_dates=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.ffill()
        return df
    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
        return None

def calculate_performance_metrics(returns_series, initial_capital=10000, risk_free_rate=0.0):
    """Calculates key performance metrics from a daily returns series."""
    if returns_series.empty or returns_series.isnull().all():
        return {
            "Final Value": f"₹{initial_capital:,.2f}", "CAGR (%)": "0.00", "Annualized Volatility (%)": "0.00",
            "Sharpe Ratio": "0.00", "Max Drawdown (%)": "0.00"
        }
    
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

def run_backtest(price_df, universe_cols, benchmark_col, lookback_period, top_n, cash_rule_params, use_pos_mom_filter):
    """The core backtest engine with dynamic universe and new filters."""
    momentum = price_df.pct_change(lookback_period)
    benchmark_sma = price_df[benchmark_col].rolling(window=cash_rule_params['sma_period']).mean()
    rebalance_dates = price_df.resample('MS').first().index # Use Month-Start frequency
    positions = pd.DataFrame(index=price_df.index, columns=universe_cols).fillna(0)
    last_positions = pd.Series(0, index=universe_cols)

    for date in price_df.index:
        if date in rebalance_dates:
            valid_universe = momentum.loc[date].dropna().index.intersection(universe_cols)
            if not valid_universe.any():
                positions.loc[date] = last_positions
                continue
            ranks = momentum.loc[date, valid_universe].rank(ascending=False)
            top_performers = ranks[ranks <= top_n].index
            filtered_top_performers = []
            if use_pos_mom_filter:
                for sector in top_performers:
                    if momentum.loc[date, sector] > 0:
                        filtered_top_performers.append(sector)
            else:
                filtered_top_performers = top_performers.tolist()
            
            invest_decision = True
            if cash_rule_params['use']:
                benchmark_mom = momentum.loc[date, benchmark_col]
                benchmark_price = price_df.loc[date, benchmark_col]
                benchmark_sma_val = benchmark_sma.loc[date]
                if pd.notna(benchmark_mom) and pd.notna(benchmark_sma_val):
                    if benchmark_mom <= 0 and benchmark_price < benchmark_sma_val:
                        invest_decision = False
            
            if invest_decision and filtered_top_performers:
                current_positions = pd.Series(0, index=universe_cols)
                current_positions[filtered_top_performers] = 1
            else:
                current_positions = pd.Series(0, index=universe_cols)
            last_positions = current_positions
        positions.loc[date] = last_positions

    daily_returns = price_df.pct_change()
    shifted_positions = positions.shift(1)
    portfolio_daily_returns = (daily_returns[universe_cols] * shifted_positions[universe_cols]).sum(axis=1)
    num_positions = shifted_positions.sum(axis=1).replace(0, 1)
    portfolio_daily_returns /= num_positions
    
    common_index = portfolio_daily_returns.dropna().index
    portfolio_daily_returns = portfolio_daily_returns.loc[common_index]
    benchmark_daily_returns = daily_returns.loc[common_index, benchmark_col]

    return portfolio_daily_returns, benchmark_daily_returns, momentum, positions

# --- Streamlit UI ---
st.title("📈 Advanced Momentum Strategy Backtester")

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
        data_filtered = full_data.loc[start_date:end_date]

        st.sidebar.header("3. Strategy Parameters")
        lookback_period = st.sidebar.slider("Momentum Lookback (days)", 5, 252, 21, 1)
        top_n = st.sidebar.slider("Sectors to Hold", 1, max(1, len(universe_cols)), min(2, len(universe_cols)), 1)
        
        st.sidebar.subheader("Filters & Rules")
        use_cash_rule = st.sidebar.checkbox("Use Hybrid Cash Rule?", value=True)
        sma_period = st.sidebar.slider("Long-term SMA for Cash Rule", 50, 252, 210, disabled=not use_cash_rule)
        use_pos_mom_filter = st.sidebar.checkbox("Use Positive Momentum Filter?", value=True)

        if st.sidebar.button("🚀 Run Backtest", type="primary"):
            cash_rule_params = {'use': use_cash_rule, 'sma_period': sma_period}
            strategy_returns, bench_returns, momentum, positions = run_backtest(
                data_filtered, universe_cols, benchmark_col, lookback_period, top_n,
                cash_rule_params, use_pos_mom_filter
            )
            st.header("📊 Performance Dashboard")
            metrics_df = pd.DataFrame({
                "Strategy": calculate_performance_metrics(strategy_returns),
                "Benchmark": calculate_performance_metrics(bench_returns)
            }).T
            st.dataframe(metrics_df, use_container_width=True)

            initial_capital = 10000
            strategy_eq = (1 + strategy_returns).cumprod() * initial_capital
            bench_eq = (1 + bench_returns).cumprod() * initial_capital
            start_day = strategy_eq.index.min() - pd.Timedelta(days=1)
            strategy_eq.loc[start_day] = initial_capital
            bench_eq.loc[start_day] = initial_capital
            strategy_eq.sort_index(inplace=True)
            bench_eq.sort_index(inplace=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=strategy_eq.index, y=strategy_eq, mode='lines', name='Strategy', line=dict(color='royalblue', width=2)))
            fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq, mode='lines', name='Benchmark', line=dict(color='grey', width=2, dash='dash')))
            fig.update_layout(title='Strategy vs. Benchmark Equity Curve', yaxis_title='Portfolio Value (Log Scale)', yaxis_type="log", legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig, use_container_width=True)
            
            tab1, tab2, tab3 = st.tabs(["Current Standings", "🗓️ Monthly Returns Analysis", "🔎 Historical Holdings"])

            with tab1:
                st.header("Current Market Snapshot")
                last_date = momentum.index[-1]
                st.write(f"Data as of: **{last_date.strftime('%d-%b-%Y')}**")
                st.subheader("Current Portfolio Holdings")
                current_positions = positions.loc[last_date]
                held_sectors = current_positions[current_positions == 1].index.tolist()
                if held_sectors:
                    st.success(f"**Holding: {', '.join(held_sectors)}**")
                else:
                    st.warning("**Holding: CASH**")
                st.subheader(f"Latest Momentum Scores ({lookback_period}-day %)")
                latest_momentum = momentum.loc[last_date, universe_cols].sort_values(ascending=False) * 100
                st.dataframe(latest_momentum.to_frame("Momentum (%)").style.format("{:.2f}%").applymap(lambda v: 'color: green' if v > 0 else 'color: red'), use_container_width=True)

            with tab2:
                st.header("Monthly Returns & Outperformance")
                monthly_data = pd.DataFrame({
                    'Strategy': strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1),
                    'Benchmark': bench_returns.resample('M').apply(lambda x: (1 + x).prod() - 1),
                })
                monthly_data['Outperformance'] = monthly_data['Strategy'] - monthly_data['Benchmark']
                
                total_months = len(monthly_data)
                outperforming_months = (monthly_data['Outperformance'] > 0).sum()
                underperforming_months = total_months - outperforming_months
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Months Analyzed", total_months)
                col2.metric("Months Outperformed Benchmark", outperforming_months, f"{outperforming_months - underperforming_months:+} months")
                col3.metric("Win Rate vs Benchmark", f"{(outperforming_months / total_months * 100):.1f}%" if total_months > 0 else "0.0%")
                monthly_data.index = monthly_data.index.strftime('%Y-%b')
                
                st.dataframe((monthly_data * 100).style.format("{:.2f}%")
                           .background_gradient(cmap='RdYlGn', subset=['Strategy', 'Benchmark'], axis=0)
                           .background_gradient(cmap='PuOr', subset=['Outperformance']),
                           use_container_width=True)

            with tab3:
                st.header("Historical Portfolio Allocations")
                # CORRECTED, ROBUST LOGIC FOR MONTHLY SNAPSHOTS
                # Resample to the start of each month ('MS'), get the first valid entry in that month,
                # and drop any months that had no trading data. This is foolproof.
                monthly_positions = positions.resample('MS').first().dropna(how='all')
                display_holdings = monthly_positions.apply(lambda row: ', '.join(row[row==1].index) or 'CASH', axis=1).to_frame("Sectors Held")
                
                # Format index for readability
                display_holdings.index = display_holdings.index.strftime('%Y-%b')
                
                st.dataframe(display_holdings.sort_index(ascending=False), use_container_width=True)
else:
    st.info("Upload an XLSX file using the sidebar to begin.")