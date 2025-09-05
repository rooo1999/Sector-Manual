# Required Libraries:
# pip install yfinance pandas pandas-ta matplotlib numpy streamlit

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# ==============================================================================
# 0. STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Nifty 50 Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide"
)

# ==============================================================================
# 1. DEFINE STRATEGY PARAMETERS & CONFIGURATION (Sidebar Inputs)
# ==============================================================================
st.sidebar.header('⚙️ Strategy Configuration')

# --- General Settings ---
ticker = st.sidebar.text_input('Ticker', '^NSEI')
start_date = st.sidebar.date_input('Start Date', datetime(2010, 1, 1))
end_date = st.sidebar.date_input('End Date', datetime.today())

# --- Strategy Mode ---
strategy_mode = st.sidebar.selectbox(
    'Strategy Mode',
    ('regime_pullback', 'regime_filter', 'ema_atr_exit', 'ema_adx', 'ema_cross'),
    index=0,
    help="""
    - **regime_pullback (Recommended):** Buy & Hold in bull markets (price > 200SMA). In bear markets, enters on pullbacks to the short EMA during a confirmed downtrend.
    - **regime_filter:** Buy & Hold in bull markets. Uses EMA/ADX/ATR strategy in bear markets.
    - **ema_atr_exit:** EMA Crossover with ADX filter for entry and an ATR Trailing Stop for exits.
    - **ema_adx:** Simple EMA Crossover with an ADX trend filter.
    - **ema_cross:** The most basic EMA Crossover strategy.
    """
)

# --- Indicator Parameters ---
st.sidebar.subheader('Indicator Parameters')
ema_short = st.sidebar.number_input('Short EMA Period', 1, 200, 21)
ema_long = st.sidebar.number_input('Long EMA Period', 1, 200, 50)
sma_regime = st.sidebar.number_input('Regime SMA Period', 50, 400, 200)
adx_threshold = st.sidebar.slider('ADX Trend Threshold', 0, 100, 25)
atr_period = st.sidebar.number_input('ATR Period', 1, 50, 14)
atr_multiplier = st.sidebar.number_input('ATR Trailing Stop Multiplier', 1.0, 5.0, 2.5, 0.1)
atr_breathing_room_multiplier = st.sidebar.number_input('ATR Breathing Room Multiplier', 1.0, 5.0, 2.5, 0.1)

# --- Backtest Parameters ---
st.sidebar.subheader('Backtest Parameters')
cash_return_rate = st.sidebar.number_input('Annual Cash Return Rate (%)', 0.0, 20.0, 6.0, 0.1) / 100
commission_rate = st.sidebar.number_input('Commission Rate per Trade (%)', 0.0, 1.0, 0.1, 0.01) / 100

# --- Compile Configuration into a Dictionary ---
CONFIG = {
    'ticker': ticker,
    'start_date': start_date.strftime('%Y-%m-%d'),
    'end_date': end_date.strftime('%Y-%m-%d'),
    'strategy_mode': strategy_mode,
    'ema_short': ema_short,
    'ema_long': ema_long,
    'sma_regime': sma_regime,
    'adx_threshold': adx_threshold,
    'atr_period': atr_period,
    'atr_multiplier': atr_multiplier,
    'atr_breathing_room_multiplier': atr_breathing_room_multiplier,
    'cash_return_rate': cash_return_rate,
    'commission_rate': commission_rate
}

# ==============================================================================
# 2. HELPER & CORE LOGIC FUNCTIONS (with Streamlit Caching)
# ==============================================================================

# Use st.cache_data to avoid re-downloading data on every interaction
@st.cache_data
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetches historical market data, ensuring a single-level column index."""
    # Fetch an extra year of data to ensure indicators are fully calculated
    start_dt = datetime.strptime(start, '%Y-%m-%d') - timedelta(days=365)
    data = yf.download(ticker, start=start_dt.strftime('%Y-%m-%d'), end=end, interval='1d', auto_adjust=True)
    if data.empty: return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data

@st.cache_data
def calculate_indicators(_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculates all necessary technical indicators."""
    df = _df.copy()
    df.ta.ema(length=config['ema_short'], append=True, col_names=('EMA_short',))
    df.ta.ema(length=config['ema_long'], append=True, col_names=('EMA_long',))
    df.ta.sma(length=config['sma_regime'], append=True, col_names=('SMA_regime',))
    df.ta.adx(length=config['atr_period'], append=True, col_names=('ADX', 'DMP', 'DMN'))
    df.ta.atr(length=config['atr_period'], append=True, col_names=('ATR',))
    df['ema_cross_up'] = ta.cross(df['EMA_short'], df['EMA_long'], above=True)
    df['ema_cross_down'] = ta.cross(df['EMA_short'], df['EMA_long'], above=False)
    
    # Trim data back to the user-specified start date after indicators are calculated
    df = df[df.index >= pd.to_datetime(config['start_date'])]
    df.dropna(inplace=True)
    return df

# The following functions are complex and depend on every parameter, so we don't cache them.
# The data fetching and indicator calculation are the slow parts, which are now cached.
def generate_signals(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generates initial entry signals based on the selected strategy mode."""
    mode = config['strategy_mode']
    entry_condition = pd.Series(False, index=df.index)

    if mode == 'ema_cross':
        entry_condition = df['ema_cross_up'] == 1
    
    elif mode in ['ema_adx', 'ema_atr_exit']:
        strong_trend = df['ADX'] > config['adx_threshold']
        entry_condition = (df['ema_cross_up'] == 1) & strong_trend
    
    elif mode == 'regime_filter':
        is_bull_regime = df['Close'] > df['SMA_regime']
        tactical_entry_condition = (df['ema_cross_up'] == 1) & (df['ADX'] > config['adx_threshold'])
        enter_bull_regime = is_bull_regime & ~is_bull_regime.shift(1).fillna(False)
        entry_condition = np.where(is_bull_regime, enter_bull_regime, tactical_entry_condition)
        
    elif mode == 'regime_pullback':
        is_bull_regime = df['Close'] > df['SMA_regime']
        trend_is_up = (df['EMA_short'] > df['EMA_long']) & (df['ADX'] > config['adx_threshold'])
        price_pulls_back_and_recovers = ta.cross(df['Close'], df['EMA_short'], above=True) == 1
        tactical_entry_condition = trend_is_up & price_pulls_back_and_recovers
        enter_bull_regime = is_bull_regime & ~is_bull_regime.shift(1).fillna(False)
        entry_condition = np.where(is_bull_regime, enter_bull_regime, tactical_entry_condition)

    df['entry_signal'] = np.where(entry_condition, 1, 0)
    return df

def run_iterative_backtest(df: pd.DataFrame, config: dict) -> (pd.DataFrame, list):
    """Iterative backtest with improved entry/exit logic and trade logging."""
    mode = config['strategy_mode']
    trades = []
    in_position = False
    entry_price, entry_date, initial_atr = 0.0, None, 0.0
    stop_loss_price = 0.0
    trailing_stop_activated = False
    
    df['position'] = 0
    df['signal'] = 0.0
    df['strategy_returns'] = 0.0
    
    daily_cash_return = (1 + config['cash_return_rate'])**(1/252) - 1
    commission = config['commission_rate']
    atr_multiplier = config['atr_multiplier']
    atr_breathing_room = config['atr_breathing_room_multiplier']
    
    for i in range(1, len(df)):
        current_date = df.index[i]
        is_bull_regime_today = df['Close'].iloc[i] > df['SMA_regime'].iloc[i]
        
        if in_position:
            trend_exit_signal = df['ema_cross_down'].iloc[i] == 1
            apply_tactical_exit = 'regime' not in mode or not is_bull_regime_today
            stop_loss_triggered = df['Low'].iloc[i] <= stop_loss_price
            
            if apply_tactical_exit and (stop_loss_triggered or trend_exit_signal):
                in_position = False
                trailing_stop_activated = False
                exit_price = stop_loss_price if stop_loss_triggered else df['Open'].iloc[i]
                df.loc[current_date, 'signal'] = -1
                df.loc[current_date, 'strategy_returns'] = (exit_price / df['Close'].iloc[i-1]) - 1 - commission
                
                trade_return = ((exit_price / entry_price) - 1) * 100
                trades.append({
                    'Entry Date': entry_date.strftime('%Y-%m-%d'), 'Entry Price': entry_price,
                    'Exit Date': current_date.strftime('%Y-%m-%d'), 'Exit Price': exit_price,
                    'Holding Period': (current_date - entry_date).days,
                    'Return Pct': trade_return - (2 * commission * 100),
                    'Exit Reason': 'Stop Loss' if stop_loss_triggered else 'EMA Cross'
                })
                continue
        
        if not in_position and df['entry_signal'].iloc[i] == 1:
            in_position = True
            entry_date, entry_price = current_date, df['Open'].iloc[i]
            initial_atr = df['ATR'].iloc[i]
            df.loc[entry_date, 'signal'] = 1
            stop_loss_price = entry_price - (initial_atr * atr_multiplier)
            trailing_stop_activated = False
            
        if in_position:
            df.loc[current_date, 'position'] = 1
            if df.loc[current_date, 'signal'] == 1: # Entry day
                 df.loc[entry_date, 'strategy_returns'] = (df['Close'].iloc[i] / entry_price) - 1 - commission
            else: # Holding day
                 df.loc[current_date, 'strategy_returns'] = df['Close'].pct_change().iloc[i]
            
            if not trailing_stop_activated and df['Close'].iloc[i] > entry_price + (initial_atr * atr_breathing_room):
                trailing_stop_activated = True
            
            if trailing_stop_activated:
                new_stop_loss = df['Close'].iloc[i] - (df['ATR'].iloc[i] * atr_multiplier)
                stop_loss_price = max(stop_loss_price, new_stop_loss)
        else:
            df.loc[current_date, 'strategy_returns'] = daily_cash_return

    return df, trades

def finalize_and_calculate_metrics(df: pd.DataFrame, benchmark_returns: pd.Series) -> (pd.DataFrame, dict):
    """Calculates cumulative returns, performance metrics, and annual returns."""
    df.dropna(subset=['strategy_returns'], inplace=True)
    if df.empty: return pd.DataFrame(), {}
    
    benchmark_returns = benchmark_returns.reindex(df.index).fillna(0)
    df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod()
    df['cumulative_market_returns'] = (1 + benchmark_returns).cumprod()

    metrics = {}
    years = (df.index[-1] - df.index[0]).days / 365.25
    if years == 0: years = 1
    
    metrics['cagr_strategy'] = ((df['cumulative_strategy_returns'].iloc[-1]) ** (1/years) - 1) * 100
    running_max = df['cumulative_strategy_returns'].cummax()
    drawdown = (df['cumulative_strategy_returns'] - running_max) / running_max
    metrics['max_drawdown_strategy'] = drawdown.min() * 100
    df['drawdown_strategy'] = drawdown
    
    metrics['cagr_market'] = ((df['cumulative_market_returns'].iloc[-1]) ** (1/years) - 1) * 100
    running_max_market = df['cumulative_market_returns'].cummax()
    drawdown_market = (df['cumulative_market_returns'] - running_max_market) / running_max_market
    metrics['max_drawdown_market'] = drawdown_market.min() * 100
    
    df['benchmark_returns'] = benchmark_returns
    def compound(series): return (1 + series).prod() - 1
    annual_returns = df[['strategy_returns', 'benchmark_returns']].resample('Y').apply(compound)
    annual_returns.index = annual_returns.index.year
    annual_returns.columns = ['Strategy', 'Benchmark']
    metrics['annual_returns'] = annual_returns * 100
    
    return df, metrics

def get_daily_signal(df: pd.DataFrame, config: dict) -> (str, str, str):
    """Determines the signal for the most recent day."""
    if df.empty or len(df) < 2:
        return "INSUFFICIENT DATA", "Cannot determine signal.", "grey"

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    signal = "HOLD"
    reason = ""
    color = "orange"

    # Check for an exit signal first
    was_in_position = prev_row['position'] == 1
    if was_in_position:
        is_bull_regime_today = last_row['Close'] > last_row['SMA_regime']
        apply_tactical_exit = 'regime' not in config['strategy_mode'] or not is_bull_regime_today
        trend_exit_signal = last_row['ema_cross_down'] == 1

        if apply_tactical_exit and trend_exit_signal:
            signal = "SELL"
            reason = f"EMA ({config['ema_short']}) crossed below EMA ({config['ema_long']})."
            color = "red"
        else:
            signal = "HOLD (IN POSITION)"
            reason = "Continue holding the existing position."
            color = "green"
    
    # If no exit, check for an entry signal
    was_in_cash = prev_row['position'] == 0
    if was_in_cash and last_row['entry_signal'] == 1:
        signal = "BUY"
        reason = "A new entry signal was generated today."
        color = "green"
    elif was_in_cash:
        signal = "HOLD (IN CASH)"
        reason = "No new entry signal. Remain in cash."
        color = "grey"
        
    return signal, reason, color

# ==============================================================================
# 3. PLOTTING FUNCTIONS
# ==============================================================================

def plot_equity_curve_and_drawdown(df: pd.DataFrame, metrics: dict):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Equity Curve
    ax1.plot(df.index, df['cumulative_market_returns'], label='Buy and Hold', color='blue')
    ax1.plot(df.index, df['cumulative_strategy_returns'], label=f'Strategy ({CONFIG["strategy_mode"]})', color='green')
    ax1.set_title('Strategy Performance vs. Buy and Hold', fontsize=16)
    ax1.set_ylabel('Cumulative Returns (1 = Initial Capital)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Drawdown
    ax2.fill_between(df.index, df['drawdown_strategy'], 0, color='red', alpha=0.3)
    ax2.plot(df.index, df['drawdown_strategy'], color='red', linewidth=1)
    ax2.set_title('Strategy Drawdown', fontsize=16)
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    fig.tight_layout()
    return fig

def plot_price_and_signals(df: pd.DataFrame, config: dict):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df['Close'], label='Price', color='skyblue', linewidth=1.5)
    ax.plot(df['EMA_short'], label=f'EMA {config["ema_short"]}', color='orange', linestyle='--', linewidth=1)
    ax.plot(df['EMA_long'], label=f'EMA {config["ema_long"]}', color='purple', linestyle='--', linewidth=1)
    ax.plot(df['SMA_regime'], label=f'SMA {config["sma_regime"]} (Regime)', color='gray', linestyle=':', linewidth=2)
    
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    ax.plot(buy_signals.index, df.loc[buy_signals.index, 'Close'], '^', markersize=10, color='green', label='Buy Signal')
    ax.plot(sell_signals.index, df.loc[sell_signals.index, 'Close'], 'v', markersize=10, color='red', label='Sell Signal')
    
    ax.set_title(f"Price, Indicators, and Signals for '{config['strategy_mode']}'", fontsize=16)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    return fig

# ==============================================================================
# 4. MAIN APP EXECUTION
# ==============================================================================

st.title(f'📈 {ticker} Trading Strategy Dashboard')
st.markdown(f"**Strategy Mode:** `{CONFIG['strategy_mode']}` | **Date Range:** `{CONFIG['start_date']}` to `{CONFIG['end_date']}`")

# --- Run the full pipeline ---
raw_data = fetch_data(CONFIG['ticker'], CONFIG['start_date'], CONFIG['end_date'])

if raw_data is not None and not raw_data.empty:
    data_with_indicators = calculate_indicators(raw_data.copy(), CONFIG)
    data_with_signals = generate_signals(data_with_indicators.copy(), CONFIG)
    
    # Note: Only iterative backtest is supported for advanced strategies
    results_df, trades_list = run_iterative_backtest(data_with_signals.copy(), CONFIG)
    
    # --- DAILY SIGNAL SECTION ---
    st.header("Today's Signal")
    
    # Get the latest price for display
    latest_price = raw_data['Close'].iloc[-1]
    price_change = latest_price - raw_data['Close'].iloc[-2]
    
    signal, reason, color = get_daily_signal(results_df, CONFIG)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"Last Close Price ({raw_data.index[-1].strftime('%d-%b-%Y')})", 
                  value=f"{latest_price:,.2f}", 
                  delta=f"{price_change:,.2f}")
    with col2:
        st.markdown(f"""
        <div style="background-color:{color}; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color:white; margin:0;">{signal}</h2>
            <p style="color:white; margin-top:5px;">{reason}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # --- BACKTEST RESULTS SECTION ---
    st.header("Backtest Performance Analysis")

    if not results_df.empty:
        benchmark_returns = raw_data['Close'].pct_change()
        final_df, metrics = finalize_and_calculate_metrics(results_df, benchmark_returns)
        
        # --- Performance Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strategy CAGR", f"{metrics.get('cagr_strategy', 0):.2f}%")
        col2.metric("Benchmark CAGR", f"{metrics.get('cagr_market', 0):.2f}%")
        col3.metric("Strategy Max Drawdown", f"{metrics.get('max_drawdown_strategy', 0):.2f}%")
        col4.metric("Benchmark Max Drawdown", f"{metrics.get('max_drawdown_market', 0):.2f}%")

        # --- Charting and Trade Logs in Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs(["Equity Curve & Drawdown", "Price Chart & Signals", "Trade Log", "Yearly Returns"])

        with tab1:
            st.pyplot(plot_equity_curve_and_drawdown(final_df, metrics))

        with tab2:
            st.pyplot(plot_price_and_signals(final_df, CONFIG))

        with tab3:
            if trades_list:
                trades_df = pd.DataFrame(trades_list)
                st.subheader("Trade Log")
                st.dataframe(trades_df.style.format({
                    'Entry Price': '{:,.2f}',
                    'Exit Price': '{:,.2f}',
                    'Return Pct': '{:.2f}%'
                }))

                # Trade Analytics
                st.subheader("Trade Analytics")
                win_rate = (trades_df['Return Pct'] > 0).mean() * 100
                avg_win = trades_df[trades_df['Return Pct'] > 0]['Return Pct'].mean()
                avg_loss = trades_df[trades_df['Return Pct'] < 0]['Return Pct'].mean()
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Trades", f"{len(trades_df)}")
                c2.metric("Win Rate", f"{win_rate:.2f}%")
                c3.metric("Payoff Ratio", f"{abs(avg_win / avg_loss) if avg_loss != 0 else '∞':.2f}")
                c4.metric("Avg Holding Period", f"{trades_df['Holding Period'].mean():.1f} days")
            else:
                st.info("No trades were executed in this backtest.")

        with tab4:
            st.subheader("Calendar Year Returns")
            st.dataframe(metrics['annual_returns'].style.format('{:,.2f}%'))
            
    else:
        st.warning("Backtest did not produce any results to analyze. Try adjusting the parameters.")

else:
    st.error(f"Could not fetch data for ticker '{CONFIG['ticker']}'. Please check the ticker symbol and date range.")

st.sidebar.info("This is a financial analysis tool. Not investment advice.")