# Required Libraries:
# pip install yfinance pandas pandas-ta matplotlib numpy

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ==============================================================================
# 1. DEFINE STRATEGY PARAMETERS & CONFIGURATION
# ==============================================================================
CONFIG = {
    'ticker': '^NSEI',
    'start_date': '2010-01-01',
    'end_date': datetime.today().strftime('%Y-%m-%d'),
    
    # --- STRATEGY MODE ---
    # 'ema_cross':    Simple EMA Crossover.
    # 'ema_adx':      EMA Crossover with ADX trend filter for entry.
    # 'ema_atr_exit': EMA Crossover with ADX filter and ATR Trailing Stop.
    # 'regime_filter': Combines Buy & Hold in bull markets (price > SMA200) 
    #                  with the advanced 'ema_atr_exit' strategy in bear markets.
    # 'regime_pullback': (NEW & RECOMMENDED) - Enters on pullbacks during established trends.
    #                    Uses the same regime filter as above.
    'strategy_mode': 'regime_pullback', # <-- Let's test the new, improved mode

    # --- INDICATOR PARAMETERS ---
    'ema_short': 21,
    'ema_long': 50,
    'sma_regime': 200,   # SMA period for the regime filter
    'adx_threshold': 25,
    'atr_period': 14,
    'atr_multiplier': 2.5,
    # --- NEW: Parameter to give trades room to breathe before trailing the stop ---
    'atr_breathing_room_multiplier': 2.5, # Stop loss starts trailing only after price moves this many ATRs in profit

    # --- BACKTEST PARAMETERS ---
    'cash_return_rate': 0.06,
    'commission_rate': 0.001
}

# ==============================================================================
# 2. HELPER & CORE LOGIC FUNCTIONS
# ==============================================================================

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetches historical market data, ensuring a single-level column index."""
    print(f"Fetching data for {ticker} from {start} to {end}...")
    data = yf.download(ticker, start=start, end=end, interval='1d', auto_adjust=True)
    if data.empty: return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    print("Data fetched successfully!")
    return data

def calculate_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculates all necessary technical indicators."""
    print(f"Calculating all indicators...")
    df.ta.ema(length=config['ema_short'], append=True, col_names=('EMA_short',))
    df.ta.ema(length=config['ema_long'], append=True, col_names=('EMA_long',))
    df.ta.sma(length=config['sma_regime'], append=True, col_names=('SMA_regime',))
    
    # --- CORRECTED LINE ---
    # We must provide names for all 3 columns that adx() creates.
    df.ta.adx(length=config['atr_period'], append=True, col_names=('ADX', 'DMP', 'DMN'))
    
    df.ta.atr(length=config['atr_period'], append=True, col_names=('ATR',))
    
    # --- MODIFIED: Pre-calculate crossover signals for cleaner logic later ---
    df['ema_cross_up'] = ta.cross(df['EMA_short'], df['EMA_long'], above=True)
    df['ema_cross_down'] = ta.cross(df['EMA_short'], df['EMA_long'], above=False)
    
    df.dropna(inplace=True)
    return df

def generate_signals(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generates initial entry signals based on the selected strategy mode."""
    mode = config['strategy_mode']
    print(f"Generating entry signals for mode: '{mode}'...")
    
    # Default to no entry
    entry_condition = pd.Series(False, index=df.index)

    # --- MODIFIED: Switched to more timely crossover events ---
    if mode == 'ema_cross':
        entry_condition = df['ema_cross_up'] == 1
    
    elif mode in ['ema_adx', 'ema_atr_exit']:
        strong_trend = df['ADX'] > config['adx_threshold']
        entry_condition = (df['ema_cross_up'] == 1) & strong_trend
    
    elif mode == 'regime_filter':
        is_bull_regime = df['Close'] > df['SMA_regime']
        tactical_entry_condition = (df['ema_cross_up'] == 1) & (df['ADX'] > config['adx_threshold'])
        # In bull regime, we want to be invested. We enter on the first day of the regime.
        enter_bull_regime = is_bull_regime & ~is_bull_regime.shift(1).fillna(False)
        entry_condition = np.where(is_bull_regime, enter_bull_regime, tactical_entry_condition)
        
    # --- NEW: Pullback Entry Strategy ---
    elif mode == 'regime_pullback':
        is_bull_regime = df['Close'] > df['SMA_regime']
        
        # Tactical entry: Wait for a pullback to the short EMA in an established uptrend
        trend_is_up = (df['EMA_short'] > df['EMA_long']) & (df['ADX'] > config['adx_threshold'])
        price_pulls_back_and_recovers = ta.cross(df['Close'], df['EMA_short'], above=True) == 1
        tactical_entry_condition = trend_is_up & price_pulls_back_and_recovers
        
        # In bull regime, we are simply long. Enter on the first day of the regime.
        enter_bull_regime = is_bull_regime & ~is_bull_regime.shift(1).fillna(False)
        entry_condition = np.where(is_bull_regime, enter_bull_regime, tactical_entry_condition)

    df['position_intended'] = np.where(entry_condition, 1, 0)
    # The 'entry_signal' will now be a simple flag (1 for entry)
    df['entry_signal'] = df['position_intended']
    return df

def run_backtest(df: pd.DataFrame, config: dict) -> (pd.DataFrame, list):
    """Runs the backtest and returns DataFrame and a list of trades."""
    # --- MODIFIED: Simplified routing to the powerful iterative backtester ---
    if config['strategy_mode'] in ['ema_atr_exit', 'regime_filter', 'regime_pullback']:
        return run_iterative_backtest(df, config)
    else:
        # Fallback for simpler strategies without complex exits
        results_df = run_vectorized_backtest(df, config)
        return results_df, []

def run_vectorized_backtest(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    print("Running vectorized backtest...")
    df['position'] = df['position_intended'].ffill().fillna(0) # Forward fill to hold position
    df['signal'] = df['position'].diff()
    daily_cash_return = (1 + config['cash_return_rate'])**(1/252) - 1
    df['strategy_returns'] = np.where(df['position'].shift(1) == 1, df['Close'].pct_change(), daily_cash_return)
    trade_days = df['signal'] != 0
    df.loc[trade_days, 'strategy_returns'] -= config['commission_rate']
    return df

def run_iterative_backtest(df: pd.DataFrame, config: dict) -> (pd.DataFrame, list):
    """Iterative backtest with improved entry/exit logic and trade logging."""
    mode = config['strategy_mode']
    print(f"Running iterative backtest for mode: '{mode}'...")
    
    trades = []
    in_position = False
    entry_price, entry_date, initial_atr = 0.0, None, 0.0
    stop_loss_price = 0.0
    trailing_stop_activated = False # --- NEW: Flag for breathing room logic ---
    
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
        
        # --- MODIFIED: EXIT LOGIC ---
        if in_position:
            # For tactical modes, add EMA cross down as an exit signal
            trend_exit_signal = df['ema_cross_down'].iloc[i] == 1
            
            # In regime modes, we only apply tactical exits during a bear market
            apply_tactical_exit = 'regime' not in mode or not is_bull_regime_today

            # Stop Loss Exit
            stop_loss_triggered = df['Low'].iloc[i] <= stop_loss_price
            
            if apply_tactical_exit and (stop_loss_triggered or trend_exit_signal):
                in_position = False
                trailing_stop_activated = False
                
                # Determine exit price based on what was triggered
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
        
        # --- MODIFIED: ENTRY LOGIC ---
        if not in_position and df['entry_signal'].iloc[i] == 1:
            in_position = True
            entry_date, entry_price = current_date, df['Open'].iloc[i]
            initial_atr = df['ATR'].iloc[i] # Store ATR at entry
            
            df.loc[entry_date, 'signal'] = 1
            df.loc[entry_date, 'position'] = 1
            
            stop_loss_price = entry_price - (initial_atr * atr_multiplier)
            trailing_stop_activated = False # Reset flag on new entry
            
            df.loc[entry_date, 'strategy_returns'] = (df['Close'].iloc[i] / entry_price) - 1 - commission
            continue
        
        # --- MODIFIED: POSITION MANAGEMENT ---
        if in_position:
            df.loc[current_date, 'position'] = 1
            df.loc[current_date, 'strategy_returns'] = df['Close'].pct_change().iloc[i]
            
            # --- NEW: Breathing Room Logic ---
            # Check if the trade is profitable enough to start trailing the stop
            if not trailing_stop_activated:
                if df['Close'].iloc[i] > entry_price + (initial_atr * atr_breathing_room):
                    trailing_stop_activated = True
                    # print(f"{current_date.date()}: Trailing stop activated for trade entered on {entry_date.date()}.") # Uncomment for debugging
            
            # Only trail the stop if it has been activated
            if trailing_stop_activated:
                new_stop_loss = df['Close'].iloc[i] - (df['ATR'].iloc[i] * atr_multiplier)
                stop_loss_price = max(stop_loss_price, new_stop_loss)
        else: # In cash
            df.loc[current_date, 'strategy_returns'] = daily_cash_return

    return df, trades

def finalize_and_calculate_metrics(df: pd.DataFrame, benchmark_returns: pd.Series, config: dict) -> (pd.DataFrame, dict):
    """Calculates cumulative returns, performance metrics, and annual returns."""
    df.dropna(subset=['strategy_returns'], inplace=True)
    if df.empty:
        print("No valid data points after cleaning. Cannot calculate metrics.")
        return pd.DataFrame(), {}
    
    benchmark_returns = benchmark_returns.reindex(df.index).fillna(0)
    
    df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod()
    df['cumulative_market_returns'] = (1 + benchmark_returns).cumprod()

    metrics = {}
    years = (df.index[-1] - df.index[0]).days / 365.25
    if years == 0: years = 1 # Avoid division by zero
    
    # CAGR and Drawdown
    metrics['cagr_strategy'] = ((df['cumulative_strategy_returns'].iloc[-1]) ** (1/years) - 1) * 100
    running_max = df['cumulative_strategy_returns'].cummax()
    drawdown = (df['cumulative_strategy_returns'] - running_max) / running_max
    metrics['max_drawdown_strategy'] = drawdown.min() * 100
    df['drawdown_strategy'] = drawdown
    
    metrics['cagr_market'] = ((df['cumulative_market_returns'].iloc[-1]) ** (1/years) - 1) * 100
    running_max_market = df['cumulative_market_returns'].cummax()
    drawdown_market = (df['cumulative_market_returns'] - running_max_market) / running_max_market
    metrics['max_drawdown_market'] = drawdown_market.min() * 100
    
    # Calculate Annual Returns
    df['benchmark_returns'] = benchmark_returns
    def compound(series): return (1 + series).prod() - 1
    annual_returns = df[['strategy_returns', 'benchmark_returns']].resample('Y').apply(compound)
    annual_returns.index = annual_returns.index.year
    annual_returns.columns = ['Strategy', 'Benchmark']
    metrics['annual_returns'] = annual_returns * 100
    
    return df, metrics

def print_results(metrics: dict, config: dict, df: pd.DataFrame, trades: list):
    """Prints the performance metrics, trade log, and annual returns."""
    print("\n--- Strategy Performance ---")
    print(f"Strategy Mode: '{config['strategy_mode']}'")
    if df.empty:
        print("No trades were executed. Cannot print results.")
        return
        
    print(f"Period: {df.index.date.min().strftime('%Y-%m-%d')} to {df.index.date.max().strftime('%Y-%m-%d')}")
    print(f"Annualized Return (CAGR): {metrics.get('cagr_strategy', 0):.2f}%")
    print(f"Max Drawdown: {metrics.get('max_drawdown_strategy', 0):.2f}%")
    
    print("\n--- Buy and Hold (Benchmark) ---")
    print(f"Annualized Return (CAGR): {metrics.get('cagr_market', 0):.2f}%")
    print(f"Max Drawdown: {metrics.get('max_drawdown_market', 0):.2f}%")

    if 'annual_returns' in metrics and not metrics['annual_returns'].empty:
        print("\n--- Calendar Year Returns ---")
        print(metrics['annual_returns'].to_string(float_format='{:,.2f}%'.format))

    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['Return Pct Num'] = trades_df['Return Pct'] # For calculations
        
        # --- NEW: More detailed trade analytics ---
        win_rate = (trades_df['Return Pct Num'] > 0).mean() * 100
        avg_win = trades_df[trades_df['Return Pct Num'] > 0]['Return Pct Num'].mean()
        avg_loss = trades_df[trades_df['Return Pct Num'] < 0]['Return Pct Num'].mean()
        avg_holding_period = trades_df['Holding Period'].mean()
        
        print(f"\n--- Trade Analytics ---")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Win: {avg_win:.2f}%")
        print(f"Average Loss: {avg_loss:.2f}%")
        print(f"Payoff Ratio (Avg Win / Avg Loss): {abs(avg_win / avg_loss):.2f}")
        print(f"Average Holding Period: {avg_holding_period:.1f} days")
        
        trades_df['Entry Price'] = trades_df['Entry Price'].map('{:,.2f}'.format)
        trades_df['Exit Price'] = trades_df['Exit Price'].map('{:,.2f}'.format)
        trades_df['Return Pct'] = trades_df['Return Pct'].map('{:.2f}%'.format)
        
        print(f"\n--- Trade Log ---")
        print(trades_df[['Entry Date', 'Entry Price', 'Exit Date', 'Exit Price', 'Holding Period', 'Return Pct', 'Exit Reason']].to_string(index=False))

def plot_price_and_signals(df: pd.DataFrame, config: dict):
    """Plots the price chart with indicators and trade signals in its own window."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df['Close'], label='Nifty 50 Price', color='skyblue', linewidth=1.5)
    ax.plot(df['EMA_short'], label=f'{config["ema_short"]}-Day EMA', color='orange', linestyle='--', linewidth=1)
    ax.plot(df['EMA_long'], label=f'{config["ema_long"]}-Day EMA', color='purple', linestyle='--', linewidth=1)
    ax.plot(df['SMA_regime'], label=f'{config["sma_regime"]}-Day SMA (Regime)', color='gray', linestyle=':', linewidth=2)
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

def plot_equity_curve(df: pd.DataFrame, metrics: dict):
    """Plots the equity curve (cumulative returns) in its own window."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df['cumulative_market_returns'], label='Buy and Hold', color='blue')
    ax.plot(df['cumulative_strategy_returns'], label='Strategy', color='green')
    ax.set_title('Strategy Performance vs. Buy and Hold', fontsize=16)
    ax.set_ylabel('Cumulative Returns (1 = Initial Capital)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if metrics:
        ax.text(0.02, 0.8, f"Strategy CAGR: {metrics['cagr_strategy']:.2f}%\nMax Drawdown: {metrics['max_drawdown_strategy']:.2f}%", 
                transform=ax.transAxes, fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    fig.tight_layout()

def plot_drawdown(df: pd.DataFrame):
    """Plots the strategy's drawdown over time in its own window."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.fill_between(df.index, df['drawdown_strategy'], 0, color='red', alpha=0.3)
    ax.plot(df.index, df['drawdown_strategy'], color='red', linewidth=1)
    ax.set_title('Strategy Drawdown', fontsize=16)
    ax.set_ylabel('Drawdown', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()

# ==============================================================================
# 4. MAIN EXECUTION SCRIPT
# ==============================================================================

def main():
    """Main function to run the full backtesting pipeline."""
    data = fetch_data(CONFIG['ticker'], CONFIG['start_date'], CONFIG['end_date'])
    
    if data is not None:
        benchmark_returns = data['Close'].pct_change()
        data_with_indicators = calculate_indicators(data.copy(), CONFIG)
        data_with_signals = generate_signals(data_with_indicators, CONFIG)
        
        results_df, trades_list = run_backtest(data_with_signals.copy(), CONFIG)
        
        if not results_df.empty:
            final_df, metrics = finalize_and_calculate_metrics(results_df, benchmark_returns, CONFIG)
            
            print_results(metrics, CONFIG, final_df, trades_list)
            
            plot_price_and_signals(final_df, CONFIG)
            plot_equity_curve(final_df, metrics)
            plot_drawdown(final_df)
            plt.show()
        else:
            print("Backtest did not produce any results to analyze or plot.")

if __name__ == '__main__':
    main()