import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
from mftool import Mftool
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
    """Reads all sheets from an Excel file, treating each as a portfolio."""
    try:
        all_sheets = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl', dtype={0: str})
        cleaned_portfolios = {}
        for sheet_name, df in all_sheets.items():
            df = df.dropna(how='all').dropna(how='all', axis=1)
            if df.empty or df.shape[1] < 2: continue
            df = df.rename(columns={df.columns[0]: 'Scheme Code'}).set_index('Scheme Code')
            numeric_cols = df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                if np.isclose(df[col].sum(), 100.0, atol=0.1): df[col] = df[col] / 100.0
            parsed_columns = {col: pd.to_datetime(col, dayfirst=False, errors='coerce') for col in df.columns}
            df = df.rename(columns=parsed_columns)
            df = df[[c for c in df.columns if isinstance(c, pd.Timestamp)]].sort_index(axis=1)
            if not df.empty: cleaned_portfolios[sheet_name] = df
        return cleaned_portfolios
    except Exception as e:
        st.error(f"Error reading Excel file: {e}"); return {}

@st.cache_data(ttl="6h")
def get_names_from_codes(scheme_codes_list):
    mf = Mftool(); all_schemes = mf.get_scheme_codes()
    return {str(code): all_schemes.get(str(code), f"Unknown: {code}") for code in scheme_codes_list}

@st.cache_data(ttl="1h", show_spinner=False)
def _fetch_navs_for_dates(portfolio_name, scheme_codes_tuple, dates_tuple):
    """Fetches NAVs only on the specific dates provided."""
    mf = Mftool()
    start_date, end_date = min(dates_tuple), max(dates_tuple)
    final_dates = pd.to_datetime(list(dates_tuple)).sort_values()
    nav_df = pd.DataFrame(index=list(scheme_codes_tuple), columns=final_dates, dtype=float)
    progress_bar = st.progress(0, text=f"Fetching NAVs for {portfolio_name}...")
    for i, code in enumerate(scheme_codes_tuple):
        try:
            history = mf.get_scheme_historical_nav(code, (start_date - pd.DateOffset(days=5)).strftime('%d-%m-%Y'), (end_date + pd.DateOffset(days=5)).strftime('%d-%m-%Y'))
            if history and 'data' in history and history['data']:
                s_df = pd.DataFrame(history['data'])
                s_df['date'] = pd.to_datetime(s_df['date'], dayfirst=True)
                s_df = s_df.set_index('date')['nav'].apply(pd.to_numeric, errors='coerce').sort_index()
                reindexed_navs = s_df.reindex(final_dates, method='ffill')
                nav_df.loc[code] = reindexed_navs.values
        except Exception as e:
            logging.warning(f"Could not fetch NAV for {code}: {e}")
        finally:
            progress_bar.progress((i + 1) / len(scheme_codes_tuple), text=f"Fetching NAVs... ({i+1}/{len(scheme_codes_tuple)})")
    progress_bar.empty(); return nav_df.dropna(axis=1, how='all')

def calculate_trailing_returns(series):
    """Calculates trailing returns, including MTD and YTD, from a periodic time series."""
    returns = {}
    periods = {'1 Month': 1, '3 Months': 3, '6 Months': 6, '1 Year': 12, '3 Years': 36, '5 Years': 60}
    series = series.sort_index().dropna()
    if len(series) < 2: return pd.Series(dtype=float)
    
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
        except (IndexError, KeyError): continue

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
        except (IndexError, KeyError): continue
        
    return pd.Series(returns)

def style_table(styler, format_str, na_rep, cmap, weight_col=None):
    """General styling function with column-wise coloring."""
    cols = list(styler.data.columns)
    if weight_col and weight_col in cols:
        styler.format({weight_col: '{:,.2%}'})
        cols.remove(weight_col)
    styler.format(format_str, subset=cols, na_rep=na_rep)
    styler.background_gradient(cmap=cmap, subset=cols, axis=0) 
    if na_rep == "None":
        styler.apply(lambda s: ['background-color: black; color: #5A5A5A;' if v == 'None' else '' for v in s], subset=cols, axis=0)
    styler.applymap_index(lambda v: 'text-align: left;', axis='index')
    return styler

# --- Streamlit UI ---
st.title("🚀 Comprehensive Portfolio Performance Dashboard")
st.markdown("Analyse portfolio performance using periodic returns based on your rebalancing dates.")

# --- MODIFICATION: Sidebar workflow changed ---
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("1. Upload Portfolio Excel File", type=['xlsx'])
    initial_investment = st.number_input("2. Initial Investment", min_value=1.0, value=10000.0, step=1000.0)

    start_date, end_date = None, None
    if uploaded_file:
        all_portfolios_data_original = read_all_portfolios(uploaded_file)
        if all_portfolios_data_original:
            all_available_dates = sorted(list(set(date for p in all_portfolios_data_original.values() for date in p.columns)))
            min_date, max_date = all_available_dates[0].date(), all_available_dates[-1].date()
            
            st.markdown("---")
            st.header("3. Set Date Range")
            start_date = st.date_input("Analysis Start Date", value=min_date, min_value=min_date, max_value=max_date)
            end_date = st.date_input("Analysis End Date", value=max_date, min_value=min_date, max_value=max_date)
    
    st.markdown("---")
    run_button = st.button("📊 Run Analysis", type="primary", use_container_width=True, disabled=not uploaded_file)

# --- MODIFICATION: Main analysis block now starts after button click with all settings ready ---
if run_button:
    if not uploaded_file:
        st.warning("⚠️ Please upload an Excel file first.")
        st.stop()
    if not start_date or not end_date:
        st.error("Could not read dates from the uploaded file. Please check the file format.")
        st.stop()
    if start_date > end_date:
        st.error("Error: End date must be on or after start date.")
        st.stop()

    # Reread from cache and filter data based on sidebar settings
    all_portfolios_data_original = read_all_portfolios(uploaded_file)
    all_portfolios_data = {}
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    for name, df in all_portfolios_data_original.items():
        filtered_df = df.loc[:, (df.columns >= start_ts) & (df.columns <= end_ts)]
        if filtered_df.shape[1] >= 2:
            all_portfolios_data[name] = filtered_df
    
    if not all_portfolios_data:
        st.warning("The selected date range does not contain enough data for analysis (requires at least two rebalancing dates).")
        st.stop()
    
    # --- Analysis proceeds as before, but on the filtered data ---
    BENCHMARKS = {
        "Nifty 50 TRI": "147794", "Nifty 500 TRI": "147625", "Smallcap 250 TRI": "147623",
        "Midcap 150 TRI": "147622", "Sensex TRI": "119597"
    }
    excel_cmap = LinearSegmentedColormap.from_list("excel_like", ["#f8696b", "#ffeb84", "#63be7b"])
    
    portfolio_results = {}
    with st.spinner("Processing all portfolios for the selected date range..."):
        all_rebalancing_dates = sorted(list(set(date for p in all_portfolios_data.values() for date in p.columns)))
        all_fund_codes = set(code for p in all_portfolios_data.values() for code in p.index)
        all_scheme_codes = tuple(all_fund_codes | set(BENCHMARKS.values()))
        
        all_periodic_navs = _fetch_navs_for_dates("All Assets", all_scheme_codes, tuple(all_rebalancing_dates))

        for name, allocations in all_portfolios_data.items():
            fund_navs = all_periodic_navs.loc[allocations.index, allocations.columns]
            if fund_navs.isnull().all().all():
                st.warning(f"Could not retrieve NAVs for portfolio '{name}'. Skipping."); continue
            
            periodic_fund_returns = fund_navs.pct_change(axis=1)
            begin_allocs = allocations.shift(1, axis=1)
            periodic_portfolio_returns = (begin_allocs * periodic_fund_returns).sum(min_count=1)
            
            value_index = (1 + periodic_portfolio_returns).cumprod().fillna(1) * initial_investment
            value_index.iloc[0] = initial_investment
            
            portfolio_results[name] = {
                'allocations': allocations,
                'periodic_fund_returns': periodic_fund_returns,
                'value_index': value_index
            }

    # --- Display Results ---
    tab_names = ["📈 Comparison"] + list(portfolio_results.keys())
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.header("Overall Portfolio Comparison")
        
        comparison_trailing_data = {}
        for name, results in portfolio_results.items():
            comparison_trailing_data[name] = calculate_trailing_returns(results['value_index'])
        
        st.subheader("Trailing Returns Comparison")
        comparison_df = pd.DataFrame(comparison_trailing_data).T
        st.dataframe(style_table(comparison_df.style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)

        st.subheader("Portfolio Value Growth Comparison")
        growth_df = pd.concat({name: res['value_index'] for name, res in portfolio_results.items()}, axis=1)
        st.line_chart(growth_df)

    for i, (name, results) in enumerate(portfolio_results.items()):
        with tabs[i+1]:
            st.header(f"Performance Analysis for: {name}")

            fund_names_map = get_names_from_codes(results['allocations'].index.tolist())
            allocations = results['allocations']
            
            benchmark_periodic_navs = all_periodic_navs.loc[list(BENCHMARKS.values()), allocations.columns]
            benchmark_periodic_navs.index = benchmark_periodic_navs.index.map({v:k for k,v in BENCHMARKS.items()})
            benchmark_periodic_returns = benchmark_periodic_navs.pct_change(axis=1)
            benchmark_value_indices = (1 + benchmark_periodic_returns).cumprod(axis=1).fillna(1) * initial_investment

            st.subheader("📈 Portfolio Growth vs Benchmarks")
            benchmark_growth_df = benchmark_value_indices.T
            combined_growth = pd.concat([results['value_index'], benchmark_growth_df], axis=1)
            combined_growth.columns.values[0] = name
            st.line_chart(combined_growth)
            st.markdown("---")
            
            st.subheader("✅ Individual Fund Performance (Trailing Returns)")
            fund_value_indices = (1 + results['periodic_fund_returns']).cumprod(axis=1).fillna(1) * initial_investment
            fund_trailing_returns = fund_value_indices.apply(calculate_trailing_returns, axis=1)
            
            fund_trailing_returns['Weight'] = allocations.iloc[:, -1]
            fund_trailing_returns.index = fund_trailing_returns.index.map(fund_names_map)
            
            cols_order = ['Weight', 'MTD', 'YTD', '1 Month', '3 Months', '6 Months', '1 Year', '3 Years', '5 Years']
            final_cols_trailing = [c for c in cols_order if c in fund_trailing_returns.columns]
            st.dataframe(style_table(fund_trailing_returns[final_cols_trailing].style, '{:.2%}', 'N/A', excel_cmap, 'Weight'), use_container_width=True)

            st.subheader("✅ Portfolio vs Benchmarks (Trailing Returns)")
            portfolio_trailing = calculate_trailing_returns(results['value_index']); portfolio_trailing.name = name
            benchmarks_trailing = benchmark_value_indices.apply(calculate_trailing_returns, axis=1)
            benchmarks_trailing.index = [f"Benchmark: {idx}" for idx in benchmarks_trailing.index]
            combined_trailing = pd.concat([portfolio_trailing.to_frame().T, benchmarks_trailing])
            final_cols_trailing_bench = [c for c in cols_order if c in combined_trailing.columns and c != 'Weight']
            st.dataframe(style_table(combined_trailing[final_cols_trailing_bench].style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)

            st.subheader("✅ Fund-wise Performance (Periodic Returns)")
            df_fund_periodic = results['periodic_fund_returns'] * 100
            df_fund_periodic.columns = [c.strftime('%b-%Y') for c in df_fund_periodic.columns]
            df_fund_periodic.index = df_fund_periodic.index.map(fund_names_map)
            df_fund_periodic['Weight'] = allocations.iloc[:, -1].values
            st.dataframe(style_table(df_fund_periodic.style, '{:.2f}', 'None', excel_cmap, 'Weight'), use_container_width=True)

            st.subheader("✅ Portfolio vs Benchmarks (Periodic Returns)")
            portfolio_periodic = results['value_index'].pct_change() * 100
            portfolio_periodic.name = f"📊 {name} Portfolio"
            benchmark_periodic = benchmark_periodic_returns * 100
            benchmark_periodic.index = [f"📊 {idx}" for idx in benchmark_periodic.index]
            combined_periodic = pd.concat([portfolio_periodic.to_frame().T, benchmark_periodic])
            combined_periodic.columns = [c.strftime('%b-%Y') for c in combined_periodic.columns]
            st.dataframe(style_table(combined_periodic.style, '{:.2f}', 'None', excel_cmap), use_container_width=True)

else:
    if not uploaded_file:
        st.info("👋 Welcome! Upload a portfolio file and click 'Run Analysis' to get started.")