import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import requests
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from dateutil.relativedelta import relativedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Comprehensive Portfolio Performance Dashboard",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state to track if the analysis has run once
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False

# --- Global Constants ---
BENCHMARKS = {
    "Nifty 50 TRI": "147794",
    "Nifty 500 TRI": "147625",
    "Smallcap 250 TRI": "147623",
    "Midcap 150 TRI": "147622",
    "Sensex TRI": "119065"
}
TRAILING_COLS_ORDER = ['MTD', 'YTD', '1 Month', '3 Months', '6 Months', '1 Year', '3 Years', '5 Years']

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# --- Helper Functions ---

@st.cache_data(ttl="1h", show_spinner="Loading portfolio allocation data...")
def read_portfolios_from_google_sheet(sheet_id):
    """Reads all sheets from a public Google Sheet and cleans them up."""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    try:
        all_sheets = pd.read_excel(url, sheet_name=None, engine='openpyxl', dtype={0: str})
        cleaned_portfolios = {}
        for sheet_name, df in all_sheets.items():
            df = df.dropna(how='all').dropna(how='all', axis=1)
            if df.empty or df.shape[1] < 2: continue
            df = df.rename(columns={df.columns[0]: 'Scheme Code'}).set_index('Scheme Code')
            numeric_cols = df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                if np.isclose(df[col].sum(), 100.0, atol=0.1):
                    df[col] = df[col] / 100.0
            parsed_columns = {col: pd.to_datetime(col, dayfirst=False, errors='coerce') for col in df.columns}
            df = df.rename(columns=parsed_columns)
            df = df[[c for c in df.columns if isinstance(c, pd.Timestamp)]].sort_index(axis=1)
            if not df.empty:
                cleaned_portfolios[sheet_name] = df
        return cleaned_portfolios
    except Exception as e:
        st.error(f"Error reading from Google Sheet. Please ensure the Sheet ID is correct and the sheet is public ('Anyone with the link'). Error: {e}")
        return {}

@st.cache_data(ttl="6h")
def get_names_from_codes(scheme_codes_list):
    """Fetches mutual fund names from a list of scheme codes using an API."""
    names = {}
    for code in scheme_codes_list:
        try:
            response = requests.get(f"https://api.mfapi.in/mf/{code}")
            if response.status_code == 200:
                scheme_name = response.json().get("meta", {}).get("scheme_name", f"Unknown: {code}")
                names[str(code)] = scheme_name
            else:
                names[str(code)] = f"Unknown: {code}"
        except Exception as e:
            logging.warning(f"Error fetching scheme name for {code}: {e}")
            names[str(code)] = f"Unknown: {code}"
    return names

@st.cache_data(ttl="1h", show_spinner="Fetching all historical NAV data...")
def _fetch_full_nav_history(scheme_codes_tuple):
    """
    Fetches the complete NAV history for a tuple of scheme codes.
    Returns a dictionary mapping scheme codes to their historical NAV DataFrame.
    """
    all_nav_history = {}
    progress_bar = st.progress(0, text=f"Fetching historical NAVs...")

    for i, code in enumerate(scheme_codes_tuple):
        try:
            url = f"https://api.mfapi.in/mf/{code}"
            response = requests.get(url)
            if response.status_code == 200:
                nav_data = response.json().get("data", [])
                if nav_data:
                    df = pd.DataFrame(nav_data)
                    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
                    df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
                    df = df.dropna().set_index('date').sort_index()
                    all_nav_history[code] = df['nav']
            else:
                logging.warning(f"Failed to fetch full NAV history for scheme {code}: HTTP {response.status_code}")
        except Exception as e:
            logging.warning(f"Exception fetching full NAV history for scheme {code}: {e}")
        finally:
            progress_bar.progress((i + 1) / len(scheme_codes_tuple), text=f"Fetching NAVs... ({i+1}/{len(scheme_codes_tuple)})")

    progress_bar.empty()
    return all_nav_history

def calculate_trailing_returns(series):
    """Calculates trailing returns using relativedelta and the financially-correct 'pad' method."""
    returns = {}
    series = series.sort_index().dropna()
    if len(series) < 2:
        return pd.Series(dtype=float)

    end_date, end_value = series.index[-1], series.iloc[-1]
    
    special_periods = {
        'MTD': datetime(end_date.year, end_date.month, 1),
        'YTD': pd.to_datetime(f'{end_date.year - 1}-12-31')
    }

    for period_name, start_date_target in special_periods.items():
        try:
            position = series.index.get_indexer([start_date_target], method='pad')[0]
            actual_start_date, start_value = series.index[position], series.iloc[position]
            if pd.notna(start_value) and start_value > 0 and actual_start_date < end_date:
                returns[period_name] = (end_value / start_value) - 1
        except (IndexError, KeyError):
            continue

    periods = {
        '1 Month': relativedelta(months=1), '3 Months': relativedelta(months=3), '6 Months': relativedelta(months=6),
        '1 Year': relativedelta(years=1), '3 Years': relativedelta(years=3), '5 Years': relativedelta(years=5)
    }
    
    for period_name, delta in periods.items():
        start_date_target = end_date - delta
        try:
            position = series.index.get_indexer([start_date_target], method='pad')[0]
            actual_start_date, start_value = series.index[position], series.iloc[position]
            
            if pd.notna(start_value) and start_value > 0 and actual_start_date < end_date:
                days_in_period = (end_date - actual_start_date).days
                if (delta.years is not None and delta.years >= 1) or \
                   (delta.months is not None and delta.months >= 12 and days_in_period > 200):
                     returns[period_name] = ((end_value / start_value) ** (365.25 / days_in_period)) - 1
                else:
                    returns[period_name] = (end_value / start_value) - 1
        except (IndexError, KeyError):
            continue
            
    return pd.Series(returns)

def style_table(styler, format_str, na_rep, cmap, weight_col=None):
    """Applies consistent styling to a DataFrame Styler object."""
    cols = list(styler.data.columns)
    if weight_col and weight_col in cols:
        styler.format({weight_col: '{:,.2%}'})
        cols.remove(weight_col)
    
    styler.format(format_str, subset=cols, na_rep=na_rep)
    styler.background_gradient(cmap=cmap, subset=cols, axis=0) 
    styler.applymap_index(lambda v: 'text-align: left;', axis='index')
    return styler

# --- Main App ---
st.title("🚀 Comprehensive Portfolio Performance Dashboard")
st.markdown("Analyse portfolio performance using periodic returns from a live data source and **up-to-date trailing returns** from market data.")

# --- Data Loading ---
all_portfolios_data_original = None
all_navs_df = None

try:
    google_sheet_id = st.secrets["GOOGLE_SHEET_ID"]
    all_portfolios_data_original = read_portfolios_from_google_sheet(google_sheet_id)
except KeyError:
    st.error("`GOOGLE_SHEET_ID` not found in Streamlit secrets. Please add it to your `.streamlit/secrets.toml` file.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during data loading: {e}")
    st.stop()

if all_portfolios_data_original:
    with st.spinner("Fetching market data..."):
        all_fund_codes = set(code for p in all_portfolios_data_original.values() for code in p.index)
        all_scheme_codes = tuple(all_fund_codes | set(BENCHMARKS.values()))
        
        full_nav_history = _fetch_full_nav_history(all_scheme_codes)
        all_navs_df = pd.DataFrame(full_nav_history).ffill().bfill()
else:
    st.warning("No portfolio data was loaded from the Google Sheet. Please check the sheet's format and sharing settings.")
    st.stop()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    initial_investment = st.number_input("1. Initial Investment", min_value=1.0, value=10000.0, step=1000.0)

    start_date, end_date = None, None
    if not all_navs_df.empty:
        api_min_date = all_navs_df.index.min().date()
        api_max_date = all_navs_df.index.max().date()
        
        default_start_date = datetime(2019, 12, 30).date()
        
        st.markdown("---")
        st.header("2. Set Date Range")
        start_date = st.date_input(
            "Analysis Start Date", 
            value=max(default_start_date, api_min_date), 
            min_value=api_min_date, 
            max_value=api_max_date
        )
        end_date = st.date_input(
            "Analysis End Date", 
            value=api_max_date, 
            min_value=api_min_date, 
            max_value=api_max_date
        )

    st.markdown("---")
    run_button = st.button("📊 Run Analysis", type="primary", use_container_width=True, disabled=(not start_date))

# --- Main Execution Block ---
if run_button or not st.session_state.analysis_run:
    if not all_portfolios_data_original or all_navs_df is None:
        st.error("Data could not be loaded. Please refresh the page.")
        st.stop()
    if start_date > end_date:
        st.error("Error: End date must be on or after start date.")
        st.stop()
    
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    
    all_portfolios_data = {}
    skipped_portfolios = []
    for name, df in all_portfolios_data_original.items():
        filtered_allocations = df.loc[:, (df.columns >= start_ts) & (df.columns <= end_ts)]
        if filtered_allocations.shape[1] >= 2:
            all_portfolios_data[name] = filtered_allocations
        elif not filtered_allocations.empty:
            skipped_portfolios.append(name)
    
    if skipped_portfolios:
        st.warning(f"The following portfolios were skipped as they have fewer than two rebalancing dates in the selected range: **{', '.join(skipped_portfolios)}**")

    if not all_portfolios_data:
        st.warning("No portfolios with sufficient data in the selected date range. Please select a wider range or check the data source.")
        st.stop()
        
    navs_df_filtered = all_navs_df.loc[start_ts:end_ts]
    
    portfolio_results = {}
    excel_cmap = LinearSegmentedColormap.from_list("excel_like", ["#f8696b", "#ffeb84", "#63be7b"])

    with st.spinner("Calculating portfolio performance..."):
        all_daily_returns = navs_df_filtered.pct_change()
        latest_date = navs_df_filtered.index.max()
        earliest_date = navs_df_filtered.index.min()
        
        for name, allocations in all_portfolios_data.items():
            portfolio_start_date = allocations.columns.min()
            date_range = pd.date_range(start=portfolio_start_date, end=latest_date, freq='D')
            portfolio_fund_returns = all_daily_returns[allocations.index].reindex(date_range).fillna(0)
            daily_target_allocations = allocations.T.reindex(date_range, method='ffill')
            daily_value_index = pd.Series(index=date_range, dtype=float)
            daily_value_index.iloc[0] = initial_investment
            holdings_value = pd.DataFrame(index=date_range, columns=allocations.index, dtype=float)
            holdings_value.iloc[0] = initial_investment * daily_target_allocations.iloc[0]
            for t in range(1, len(date_range)):
                prev_date, current_date = date_range[t-1], date_range[t]
                grown_holdings = holdings_value.loc[prev_date] * (1 + portfolio_fund_returns.loc[current_date])
                if not daily_target_allocations.loc[current_date].equals(daily_target_allocations.loc[prev_date]):
                    total_portfolio_value = grown_holdings.sum()
                    holdings_value.loc[current_date] = total_portfolio_value * daily_target_allocations.loc[current_date]
                else:
                    holdings_value.loc[current_date] = grown_holdings
                daily_value_index.loc[current_date] = holdings_value.loc[current_date].sum()
            daily_value_index = daily_value_index.dropna()

            # --- Periodic Returns Calculation (for individual tab) ---
            rebal_dates = allocations.columns
            periodic_navs = navs_df_filtered.reindex(rebal_dates, method='ffill')
            periodic_fund_returns = periodic_navs.loc[:, allocations.index].pct_change()
            begin_allocs = allocations.shift(1, axis=1)
            periodic_portfolio_returns = (begin_allocs.T * periodic_fund_returns).sum(axis=1, min_count=1)
            benchmark_periodic_returns = periodic_navs.loc[:, BENCHMARKS.values()].pct_change()
            benchmark_periodic_returns.columns = BENCHMARKS.keys()
            
            # --- Trailing Fund Returns Calculation (for individual tab) ---
            fund_daily_indices = (1 + portfolio_fund_returns).cumprod() * initial_investment
            fund_trailing_returns = fund_daily_indices.apply(calculate_trailing_returns, axis=0).T
            
            # --- Store Results for each portfolio ---
            portfolio_results[name] = {
                'allocations': allocations, 
                'daily_value_index': daily_value_index,
                'periodic_fund_returns': periodic_fund_returns.T, 
                'periodic_portfolio_returns': periodic_portfolio_returns, 
                'benchmark_periodic_returns': benchmark_periodic_returns,
                'fund_trailing_returns': fund_trailing_returns
            }
        
        # --- UNIFIED & CONSISTENT CALCULATIONS (THE SINGLE SOURCE OF TRUTH) ---
        # 1. Combine all daily portfolio values into one DataFrame
        all_portfolio_indices = pd.concat({name: res['daily_value_index'] for name, res in portfolio_results.items()}, axis=1)

        # 2. Create a single DataFrame for all benchmark daily values
        benchmark_daily_indices = pd.DataFrame()
        unified_date_range = pd.date_range(start=earliest_date, end=latest_date, freq='D')
        for b_name, b_code in BENCHMARKS.items():
            if b_code in all_daily_returns:
                b_returns = all_daily_returns[b_code].reindex(unified_date_range).fillna(0)
                benchmark_daily_indices[b_name] = (1 + b_returns).cumprod().fillna(1) * initial_investment

        # 3. Calculate all Trailing Returns from the unified data
        portfolios_trailing_df = all_portfolio_indices.apply(calculate_trailing_returns).T
        benchmarks_trailing_df = benchmark_daily_indices.apply(calculate_trailing_returns).T

        # 4. Calculate all YOY Returns from the unified data
        portfolios_yoy_df = all_portfolio_indices.resample('YE').last().pct_change().iloc[1:]
        benchmarks_yoy_df = benchmark_daily_indices.resample('YE').last().pct_change().iloc[1:]
        
        # 5. Calculate all MOM Returns from the unified data
        portfolios_mom_df = all_portfolio_indices.resample('ME').last().pct_change().iloc[1:, :]
        benchmarks_mom_df = benchmark_daily_indices.resample('ME').last().pct_change().iloc[1:, :]


    # --- UI Rendering ---
    tab_names = ["📈 Comparison"] + list(portfolio_results.keys())
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.header("Overall Portfolio Comparison")

        # --- Trailing Returns ---
        st.subheader("Trailing Returns: Portfolios")
        final_cols_trailing = [c for c in TRAILING_COLS_ORDER if c in portfolios_trailing_df.columns]
        st.dataframe(style_table(portfolios_trailing_df[final_cols_trailing].style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)

        st.subheader("Trailing Returns: Benchmarks")
        final_cols_bench_trailing = [c for c in TRAILING_COLS_ORDER if c in benchmarks_trailing_df.columns]
        st.dataframe(style_table(benchmarks_trailing_df[final_cols_bench_trailing].style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)
        st.markdown("---")

        # --- Year-on-Year Returns ---
        st.subheader("Calendar Year (YOY) Performance")
        st.markdown("##### **Portfolio Comparison (YOY)**")
        if not portfolios_yoy_df.empty:
            portfolios_yoy_df.index = portfolios_yoy_df.index.strftime('%Y')
            st.dataframe(style_table(portfolios_yoy_df.T.style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)
        else:
            st.info("Not enough data for a full year-on-year portfolio comparison.")
            
        st.markdown("##### **Benchmark Comparison (YOY)**")
        if not benchmarks_yoy_df.empty:
            benchmarks_yoy_df.index = benchmarks_yoy_df.index.strftime('%Y')
            st.dataframe(style_table(benchmarks_yoy_df.T.style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)
        else:
            st.info("Not enough data for a full year-on-year benchmark comparison.")
        st.markdown("---")

        # --- Month-on-Month Returns ---
        st.subheader("Monthly (MOM) Performance (Last 12 Months)")
        st.markdown("##### **Portfolio Comparison (MOM)**")
        if not portfolios_mom_df.empty:
            portfolios_mom_df.index = portfolios_mom_df.index.strftime('%b-%Y')
            st.dataframe(style_table(portfolios_mom_df.T.iloc[:, -12:].style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)
        else:
            st.info("Not enough data for a month-on-month portfolio comparison.")

        st.markdown("##### **Benchmark Comparison (MOM)**")
        if not benchmarks_mom_df.empty:
            benchmarks_mom_df.index = benchmarks_mom_df.index.strftime('%b-%Y')
            st.dataframe(style_table(benchmarks_mom_df.T.iloc[:, -12:].style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)
        else:
            st.info("Not enough data for a month-on-month benchmark comparison.")


    for i, (name, results) in enumerate(portfolio_results.items()):
        with tabs[i+1]:
            st.header(f"Performance Analysis for: {name}")
            fund_names_map = get_names_from_codes(results['allocations'].index.tolist())
            
            st.subheader("✅ Performance (Trailing Returns)")
            st.markdown("##### **Individual Funds**")
            fund_trailing_returns_display = results['fund_trailing_returns'].copy()
            fund_trailing_returns_display['Weight'] = results['allocations'].iloc[:, -1]
            fund_trailing_returns_display.index = fund_trailing_returns_display.index.map(fund_names_map)
            final_cols_trailing_funds = ['Weight'] + [c for c in TRAILING_COLS_ORDER if c in fund_trailing_returns_display.columns]
            st.dataframe(style_table(fund_trailing_returns_display[final_cols_trailing_funds].style, '{:.2%}', 'N/A', excel_cmap, 'Weight'), use_container_width=True)
            
            st.markdown("##### **Portfolio vs. Benchmarks**")
            # Combine this portfolio's trailing returns with the consistent, globally calculated benchmark returns
            portfolio_trailing = portfolios_trailing_df.loc[[name]]
            combined_trailing = pd.concat([portfolio_trailing, benchmarks_trailing_df])
            st.dataframe(style_table(combined_trailing[final_cols_trailing].style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)

            st.markdown("---")
            st.subheader("📅 Calendar Year Performance (Year-on-Year)")
            st.markdown("##### **Portfolio vs. Benchmarks**")
            # Combine this portfolio's YOY returns with the consistent, globally calculated benchmark returns
            portfolio_yoy = portfolios_yoy_df[[name]].T
            portfolio_yoy.index.name = name
            combined_yoy = pd.concat([portfolio_yoy, benchmarks_yoy_df.T])
            
            if not combined_yoy.empty and not combined_yoy.columns.empty:
                 st.dataframe(style_table(combined_yoy.style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)
            else:
                st.info("Not enough data for a full year-on-year comparison for Portfolio vs. Benchmarks.")

            st.markdown("---")
            st.subheader("✅ Performance Between Rebalancing Dates (Periodic Returns)")
            st.info("This section shows the returns calculated between the specific dates in your uploaded file.")
            
            st.markdown("##### **Individual Funds (Periodic)**")
            df_fund_periodic = results['periodic_fund_returns'].mul(100)
            df_fund_periodic.columns = [c.strftime('%b-%Y') for c in df_fund_periodic.columns]
            df_fund_periodic.index = df_fund_periodic.index.map(fund_names_map)
            df_fund_periodic['Weight'] = results['allocations'].iloc[:, -1].values
            st.dataframe(style_table(df_fund_periodic.style, '{:.2f}%', 'None', excel_cmap, 'Weight'), use_container_width=True)

            st.markdown("##### **Portfolio vs. Benchmarks (Periodic)**")
            portfolio_periodic = results['periodic_portfolio_returns'].mul(100)
            portfolio_periodic.name = f"📊 {name} Portfolio"
            benchmark_periodic = results['benchmark_periodic_returns'].T.mul(100)
            combined_periodic = pd.concat([portfolio_periodic.to_frame().T, benchmark_periodic])
            combined_periodic.columns = [c.strftime('%b-%Y') for c in combined_periodic.columns]
            st.dataframe(style_table(combined_periodic.style, '{:.2f}%', 'None', excel_cmap), use_container_width=True)
            
    # Set the flag to True after the first successful run.
    st.session_state.analysis_run = True


elif not all_portfolios_data_original:
    st.info("👋 Welcome! Data is being loaded. If you see an error, please check the secrets configuration.")