import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import requests
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap

# --- Page Configuration ---
st.set_page_config(
    page_title="Comprehensive Portfolio Performance Dashboard",
    page_icon="🚀",
    layout="wide"
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# --- Helper Functions ---
@st.cache_data
def read_all_portfolios(uploaded_file):
    """Reads all sheets from an Excel file and cleans them up to be portfolio allocation DataFrames."""
    try:
        all_sheets = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl', dtype={0: str})
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
        st.error(f"Error reading Excel file: {e}")
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
    """Calculates trailing returns using pandas DateOffset and 'nearest' date matching."""
    returns = {}
    periods = {'1 Month': 1, '3 Months': 3, '6 Months': 6, '1 Year': 12, '3 Years': 36, '5 Years': 60}
    series = series.sort_index().dropna()
    if len(series) < 2: 
        return pd.Series(dtype=float)

    end_date, end_value = series.index[-1], series.iloc[-1]

    special_periods = {
        'MTD': end_date - pd.offsets.MonthBegin(1),
        'YTD': end_date - pd.offsets.YearBegin(1)
    }

    for period_name, start_date_target in special_periods.items():
        try:
            position = series.index.get_indexer([start_date_target], method='nearest')[0]
            actual_start_date, start_value = series.index[position], series.iloc[position]
            if pd.notna(start_value) and start_value > 0 and actual_start_date < end_date:
                returns[period_name] = (end_value / start_value) - 1
        except (IndexError, KeyError): 
            continue

    for period_name, months in periods.items():
        start_date_target = end_date - pd.DateOffset(months=months)
        try:
            position = series.index.get_indexer([start_date_target], method='nearest')[0]
            actual_start_date, start_value = series.index[position], series.iloc[position]
            if pd.notna(start_value) and start_value > 0 and actual_start_date < end_date:
                days_in_period = (end_date - actual_start_date).days
                if months >= 12 and days_in_period > 200:
                    returns[period_name] = ((end_value / start_value) ** (365.25 / days_in_period)) - 1
                else:
                    returns[period_name] = (end_value / start_value) - 1
        except (IndexError, KeyError): 
            continue
            
    return pd.Series(returns)

def display_trailing_date_ranges(series):
    """Displays actual start and end dates used for each trailing return calculation, matching the 'nearest' logic."""
    series = series.sort_index().dropna()
    if len(series) < 2:
        return

    end_date = series.index[-1]
    st.markdown("### 📆 Trailing Return Date Ranges (Based on latest available data)")
    st.write(f"**Calculation End Date:** {end_date.date()}")
    
    special_periods = {
        'MTD': end_date - pd.offsets.MonthBegin(1),
        'YTD': end_date - pd.offsets.YearBegin(1)
    }

    for name, target_start in special_periods.items():
        try:
            pos = series.index.get_indexer([target_start], method='nearest')[0]
            actual_start = series.index[pos]
            st.write(f"• **{name}:** Actual Start {actual_start.date()} (Target: {target_start.date()})")
        except IndexError:
            continue
            
    periods = {'1 Month': 1, '3 Months': 3, '6 Months': 6, '1 Year': 12, '3 Years': 36, '5 Years': 60}
    for name, months in periods.items():
        target_start = end_date - pd.DateOffset(months=months)
        try:
            pos = series.index.get_indexer([target_start], method='nearest')[0]
            actual_start = series.index[pos]
            st.write(f"• **{name}:** Actual Start {actual_start.date()} (Target: {target_start.date()})")
        except IndexError:
            continue

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

# --- Streamlit UI ---
st.title("🚀 Comprehensive Portfolio Performance Dashboard")
st.markdown("Analyse portfolio performance using periodic returns from your file and **up-to-date trailing returns** from live market data.")

with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("1. Upload Portfolio Excel File", type=['xlsx'])
    initial_investment = st.number_input("2. Initial Investment", min_value=1.0, value=10000.0, step=1000.0)
    st.markdown("---")
    run_button = st.button("📊 Run Analysis", type="primary", use_container_width=True, disabled=not uploaded_file)

if run_button:
    if not uploaded_file:
        st.warning("⚠️ Please upload an Excel file first.")
        st.stop()

    all_portfolios_data = read_all_portfolios(uploaded_file)
    if not all_portfolios_data:
        st.error("Could not read any valid portfolios from the uploaded file. Please check the file format.")
        st.stop()

    BENCHMARKS = {
        "Nifty 50 TRI": "120716", "Nifty 500 TRI": "153161", "Smallcap 250 TRI": "153233",
        "Midcap 150 TRI": "153089", "Sensex TRI": "149803"
    }
    excel_cmap = LinearSegmentedColormap.from_list("excel_like", ["#f8696b", "#ffeb84", "#63be7b"])

    portfolio_results = {}
    
    with st.spinner("Processing all portfolios... This may take a moment."):
        all_fund_codes = set(code for p in all_portfolios_data.values() for code in p.index)
        all_scheme_codes = tuple(all_fund_codes | set(BENCHMARKS.values()))
        full_nav_history = _fetch_full_nav_history(all_scheme_codes)

        all_navs_df = pd.DataFrame(full_nav_history).ffill().bfill()
        if all_navs_df.empty:
            st.error("Could not fetch any NAV data. Please check the scheme codes or try again later.")
            st.stop()

        all_daily_returns = all_navs_df.pct_change()
        latest_date = all_navs_df.index.max()
        
        earliest_date = min(p.columns.min() for p in all_portfolios_data.values())

        for name, allocations in all_portfolios_data.items():
            start_date = allocations.columns.min()
            
            # --- Daily Value Index Calculation (for Trailing Returns & Growth Chart) ---
            date_range = pd.date_range(start=start_date, end=latest_date, freq='D')
            daily_allocations = allocations.T.reindex(date_range, method='ffill')
            
            portfolio_fund_returns_subset = all_daily_returns[allocations.index]
            aligned_fund_returns = portfolio_fund_returns_subset.reindex(daily_allocations.index)
            daily_portfolio_returns = (daily_allocations * aligned_fund_returns).sum(axis=1, min_count=1)
            daily_portfolio_returns.fillna(0, inplace=True)

            daily_value_index = (1 + daily_portfolio_returns).cumprod().fillna(1) * initial_investment
            daily_value_index.iloc[0] = initial_investment

            # --- FIX: Periodic Returns Calculation (Reinstating the original, correct logic) ---
            rebal_dates = allocations.columns
            # 1. Create a NAV table for the specific rebalancing dates, using ffill to handle non-trading days.
            periodic_navs = all_navs_df.reindex(rebal_dates, method='ffill')
            
            # 2. Calculate periodic returns for all funds on this corrected NAV table.
            periodic_fund_returns = periodic_navs.loc[:, allocations.index].pct_change()

            # 3. Use allocations from the beginning of the period to calculate portfolio returns.
            begin_allocs = allocations.shift(1, axis=1)
            periodic_portfolio_returns = (begin_allocs.T * periodic_fund_returns).sum(axis=1, min_count=1)
            
            # 4. Get benchmark returns from the same correctly-aligned NAV table.
            benchmark_periodic_returns = periodic_navs.loc[:, BENCHMARKS.values()].pct_change()
            benchmark_periodic_returns.columns = BENCHMARKS.keys()
            # --- END OF FIX ---
            
            portfolio_results[name] = {
                'allocations': allocations,
                'daily_value_index': daily_value_index,
                'periodic_fund_returns': periodic_fund_returns.T, # Transpose for fund-wise table 
                'periodic_portfolio_returns': periodic_portfolio_returns,
                'benchmark_periodic_returns': benchmark_periodic_returns
            }

        benchmark_daily_indices = {}
        unified_date_range = pd.date_range(start=earliest_date, end=latest_date, freq='D')
        for b_name, b_code in BENCHMARKS.items():
            if b_code in all_daily_returns:
                b_returns = all_daily_returns[b_code].reindex(unified_date_range).fillna(0)
                b_index = (1 + b_returns).cumprod().fillna(1) * initial_investment
                benchmark_daily_indices[b_name] = b_index

    # --- UI Rendering ---
    tab_names = ["📈 Comparison"] + list(portfolio_results.keys())
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.header("Overall Portfolio Comparison")
        st.subheader("Trailing Returns Comparison (Based on Latest Data)")
        comparison_trailing_data = {name: calculate_trailing_returns(res['daily_value_index']) for name, res in portfolio_results.items()}
        comparison_df = pd.DataFrame(comparison_trailing_data).T
        st.dataframe(style_table(comparison_df.style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)

        st.subheader("Portfolio Value Growth Comparison")
        growth_df = pd.concat({name: res['daily_value_index'] for name, res in portfolio_results.items()}, axis=1)
        st.line_chart(growth_df)

    for i, (name, results) in enumerate(portfolio_results.items()):
        with tabs[i+1]:
            st.header(f"Performance Analysis for: {name}")
            fund_names_map = get_names_from_codes(results['allocations'].index.tolist())

            st.subheader("📈 Portfolio Growth vs Benchmarks")
            portfolio_start_date = results['daily_value_index'].index.min()
            filtered_benchmark_indices = {b_name: b_index.loc[portfolio_start_date:] for b_name, b_index in benchmark_daily_indices.items()}
            benchmark_growth_df = pd.DataFrame(filtered_benchmark_indices)

            combined_growth = pd.concat([results['daily_value_index'], benchmark_growth_df], axis=1)
            combined_growth.columns.values[0] = name
            st.line_chart(combined_growth)

            st.markdown("---")
            st.subheader("✅ Performance (Trailing Returns)")
            display_trailing_date_ranges(results['daily_value_index'])

            st.markdown("##### **Portfolio vs. Benchmarks**")
            portfolio_trailing = calculate_trailing_returns(results['daily_value_index'])
            portfolio_trailing.name = name
            benchmarks_trailing = pd.DataFrame(filtered_benchmark_indices).apply(calculate_trailing_returns).T
            combined_trailing = pd.concat([portfolio_trailing.to_frame().T, benchmarks_trailing])
            
            cols_order = ['MTD', 'YTD', '1 Month', '3 Months', '6 Months', '1 Year', '3 Years', '5 Years']
            final_cols_trailing = [c for c in cols_order if c in combined_trailing.columns]
            st.dataframe(style_table(combined_trailing[final_cols_trailing].style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)

            st.markdown("##### **Individual Funds**")
            fund_daily_returns = all_daily_returns[results['allocations'].index].reindex(results['daily_value_index'].index).fillna(0)
            fund_daily_indices = (1 + fund_daily_returns).cumprod() * initial_investment
            
            fund_trailing_returns = fund_daily_indices.apply(calculate_trailing_returns, axis=0).T
            
            fund_trailing_returns['Weight'] = results['allocations'].iloc[:, -1]
            fund_trailing_returns.index = fund_trailing_returns.index.map(fund_names_map)

            final_cols_trailing_funds = ['Weight'] + [c for c in cols_order if c in fund_trailing_returns.columns]
            st.dataframe(style_table(fund_trailing_returns[final_cols_trailing_funds].style, '{:.2%}', 'N/A', excel_cmap, 'Weight'), use_container_width=True)

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
            # Use the correctly calculated periodic returns
            portfolio_periodic = results['periodic_portfolio_returns'].mul(100)
            portfolio_periodic.name = f"📊 {name} Portfolio"
            benchmark_periodic = results['benchmark_periodic_returns'].T.mul(100)
            
            combined_periodic = pd.concat([portfolio_periodic.to_frame().T, benchmark_periodic])
            combined_periodic.columns = [c.strftime('%b-%Y') for c in combined_periodic.columns]
            st.dataframe(style_table(combined_periodic.style, '{:.2f}%', 'None', excel_cmap), use_container_width=True)

else:
    if not uploaded_file:
        st.info("👋 Welcome! Upload a portfolio file and click 'Run Analysis' to get started.")