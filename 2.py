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

# --- Global Constants ---
BENCHMARKS = {
    "Nifty 50 TRI": "147794",
    "Nifty 500 TRI": "147625",
    "Smallcap 250 TRI": "147623",
    "Midcap 150 TRI": "147622",
    "Sensex TRI": "119597"
}
TRAILING_COLS_ORDER = ['MTD', 'YTD', '1 Month', '3 Months', '6 Months', '1 Year', '3 Years', '5 Years']

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# --- Helper Functions ---

# MODIFIED: New function to read from a public Google Sheet
@st.cache_data(ttl="1h", show_spinner="Loading portfolio allocation data...")
def read_portfolios_from_google_sheet(sheet_id):
    """Reads all sheets from a public Google Sheet and cleans them up."""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    try:
        # The logic inside is identical to the old function, just the source changes.
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
    # Use the new function with the ID from secrets
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
        
        st.markdown("---")
        st.header("2. Set Date Range")
        start_date = st.date_input("Analysis Start Date", value=api_min_date, min_value=api_min_date, max_value=api_max_date)
        end_date = st.date_input("Analysis End Date", value=api_max_date, min_value=api_min_date, max_value=api_max_date)

    st.markdown("---")
    run_button = st.button("📊 Run Analysis", type="primary", use_container_width=True, disabled=(not start_date))

# --- Main Execution Block ---
if run_button:
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
        # (The entire calculation logic from here remains unchanged)
        # ... [The rest of the calculation code is identical to the previous version] ...
        all_daily_returns = navs_df_filtered.pct_change()
        latest_date = navs_df_filtered.index.max()
        earliest_date = navs_df_filtered.index.min()
        
        for name, allocations in all_portfolios_data.items():
            portfolio_start_date = allocations.columns.min()
            
            date_range = pd.date_range(start=portfolio_start_date, end=latest_date, freq='D')
            daily_allocations = allocations.T.reindex(date_range, method='ffill')
            portfolio_fund_returns = all_daily_returns[allocations.index].reindex(daily_allocations.index).fillna(0)
            daily_portfolio_returns = (daily_allocations * portfolio_fund_returns).sum(axis=1, min_count=1)
            daily_value_index = (1 + daily_portfolio_returns).cumprod().fillna(1) * initial_investment
            daily_value_index.iloc[0] = initial_investment

            portfolio_trailing_returns = calculate_trailing_returns(daily_value_index)

            rebal_dates = allocations.columns
            periodic_navs = navs_df_filtered.reindex(rebal_dates, method='ffill')
            periodic_fund_returns = periodic_navs.loc[:, allocations.index].pct_change()
            begin_allocs = allocations.shift(1, axis=1)
            periodic_portfolio_returns = (begin_allocs.T * periodic_fund_returns).sum(axis=1, min_count=1)
            benchmark_periodic_returns = periodic_navs.loc[:, BENCHMARKS.values()].pct_change()
            benchmark_periodic_returns.columns = BENCHMARKS.keys()
            
            fund_daily_indices = (1 + portfolio_fund_returns).cumprod() * initial_investment
            fund_trailing_returns = fund_daily_indices.apply(calculate_trailing_returns, axis=0).T
            
            years_in_range = range(earliest_date.year, latest_date.year + 1)
            year_end_targets = [pd.to_datetime(f'{year}-12-31') for year in years_in_range]
            current_year = latest_date.year
            if current_year in years_in_range and latest_date.strftime('%Y-%m-%d') != f'{current_year}-12-31':
                current_year_index = list(years_in_range).index(current_year)
                year_end_targets[current_year_index] = latest_date
            year_end_targets = sorted(list(set(year_end_targets)))
            
            yoy_navs = navs_df_filtered.reindex(year_end_targets, method='pad').dropna(how='all')
            yoy_fund_returns = yoy_navs[allocations.index].pct_change()
            
            yoy_allocations = daily_allocations.reindex(yoy_navs.index, method='pad')
            begin_year_allocs = yoy_allocations.shift(1)
            
            yoy_portfolio_returns = (yoy_fund_returns * begin_year_allocs).sum(axis=1, min_count=1)
            
            yoy_benchmark_returns = yoy_navs[list(BENCHMARKS.values())].pct_change()
            yoy_benchmark_returns.columns = BENCHMARKS.keys()
            
            portfolio_results[name] = {
                'allocations': allocations,
                'daily_value_index': daily_value_index,
                'portfolio_trailing_returns': portfolio_trailing_returns,
                'periodic_fund_returns': periodic_fund_returns.T,
                'periodic_portfolio_returns': periodic_portfolio_returns,
                'benchmark_periodic_returns': benchmark_periodic_returns,
                'fund_trailing_returns': fund_trailing_returns,
                'yoy_fund_returns': yoy_fund_returns,
                'yoy_portfolio_returns': yoy_portfolio_returns,
                'yoy_benchmark_returns': yoy_benchmark_returns
            }

        benchmark_daily_indices = {}
        unified_date_range = pd.date_range(start=earliest_date, end=latest_date, freq='D')
        for b_name, b_code in BENCHMARKS.items():
            if b_code in all_daily_returns:
                b_returns = all_daily_returns[b_code].reindex(unified_date_range).fillna(0)
                b_index = (1 + b_returns).cumprod().fillna(1) * initial_investment
                benchmark_daily_indices[b_name] = b_index
    
    # --- UI Rendering ---
    # (The entire UI rendering logic from here remains unchanged)
    # ... [The rest of the UI rendering code is identical to the previous version] ...
    tab_names = ["📈 Comparison"] + list(portfolio_results.keys())
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.header("Overall Portfolio Comparison")
        st.subheader("Trailing Returns Comparison")
        
        comparison_df = pd.DataFrame({
            name: res['portfolio_trailing_returns'] 
            for name, res in portfolio_results.items()
        }).T
        
        final_cols_comparison = [c for c in TRAILING_COLS_ORDER if c in comparison_df.columns]
        st.dataframe(style_table(comparison_df[final_cols_comparison].style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)

        st.subheader("Portfolio Value Growth Comparison")
        growth_df = pd.concat({name: res['daily_value_index'] for name, res in portfolio_results.items()}, axis=1)
        st.line_chart(growth_df)

    for i, (name, results) in enumerate(portfolio_results.items()):
        with tabs[i+1]:
            st.header(f"Performance Analysis for: {name}")
            fund_names_map = get_names_from_codes(results['allocations'].index.tolist())

            st.subheader("📈 Portfolio Growth vs Benchmarks")
            portfolio_start_date_tab = results['daily_value_index'].index.min()
            portfolio_start_value = results['daily_value_index'].iloc[0]
            
            filtered_benchmark_indices = {
                b_name: b_index.loc[portfolio_start_date_tab:] 
                for b_name, b_index in benchmark_daily_indices.items()
            }
            
            normalized_benchmark_indices = {}
            for b_name, b_series in filtered_benchmark_indices.items():
                if not b_series.empty:
                    normalized_benchmark_indices[b_name] = (b_series / b_series.iloc[0]) * portfolio_start_value
            
            benchmark_growth_df = pd.DataFrame(normalized_benchmark_indices)
            combined_growth = pd.concat([results['daily_value_index'], benchmark_growth_df], axis=1)
            combined_growth.columns.values[0] = name
            st.line_chart(combined_growth)

            st.markdown("---")
            st.subheader("✅ Performance (Trailing Returns)")
            st.markdown("##### **Individual Funds**")
            fund_trailing_returns_display = results['fund_trailing_returns'].copy()
            fund_trailing_returns_display['Weight'] = results['allocations'].iloc[:, -1]
            fund_trailing_returns_display.index = fund_trailing_returns_display.index.map(fund_names_map)
            
            final_cols_trailing_funds = ['Weight'] + [c for c in TRAILING_COLS_ORDER if c in fund_trailing_returns_display.columns]
            st.dataframe(style_table(fund_trailing_returns_display[final_cols_trailing_funds].style, '{:.2%}', 'N/A', excel_cmap, 'Weight'), use_container_width=True)
            
            st.markdown("##### **Portfolio vs. Benchmarks**")
            portfolio_trailing_returns = results['portfolio_trailing_returns']
            portfolio_trailing_returns.name = name
            
            benchmarks_trailing = pd.DataFrame(filtered_benchmark_indices).apply(calculate_trailing_returns).T
            combined_trailing = pd.concat([portfolio_trailing_returns.to_frame().T, benchmarks_trailing])
            
            final_cols_trailing = [c for c in TRAILING_COLS_ORDER if c in combined_trailing.columns]
            st.dataframe(style_table(combined_trailing[final_cols_trailing].style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)

            st.markdown("---")
            st.subheader("📅 Calendar Year Performance (Year-on-Year)")
            
            st.markdown("##### **Individual Funds**")
            yoy_fund_df = results['yoy_fund_returns'].T.mul(100)
            if not yoy_fund_df.empty:
                yoy_fund_df.index = yoy_fund_df.index.map(fund_names_map)
                yoy_fund_df.columns = [c.strftime('%Y') for c in yoy_fund_df.columns]
                yoy_fund_df = yoy_fund_df.iloc[:, 1:]
            
            if not yoy_fund_df.empty and not yoy_fund_df.columns.empty:
                st.dataframe(style_table(yoy_fund_df.style, '{:.2f}%', 'N/A', excel_cmap), use_container_width=True)
            else:
                st.info("Not enough data for a full year-on-year comparison for individual funds.")

            st.markdown("##### **Portfolio vs. Benchmarks**")
            yoy_portfolio = results['yoy_portfolio_returns'].mul(100)
            yoy_portfolio.name = f"📊 {name} Portfolio"
            yoy_benchmarks = results['yoy_benchmark_returns'].T.mul(100)
            combined_yoy = pd.concat([yoy_portfolio.to_frame().T, yoy_benchmarks])
            
            if not combined_yoy.empty:
                combined_yoy.columns = [c.strftime('%Y') for c in combined_yoy.columns]
                combined_yoy = combined_yoy.iloc[:, 1:]
            
            if not combined_yoy.empty and not combined_yoy.columns.empty:
                st.dataframe(style_table(combined_yoy.style, '{:.2f}%', 'N/A', excel_cmap), use_container_width=True)
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

elif not all_portfolios_data_original:
    # This block will now only be reached if the initial load from Google Sheets fails.
    st.info("👋 Welcome! Data is being loaded. If you see an error, please check the secrets configuration.")