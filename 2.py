import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import requests
from datetime import datetime, date
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
    "Sensex TRI": "119065"
}
TRAILING_COLS_ORDER = ['MTD', 'YTD', '1 Month', '3 Months', '6 Months', '1 Year', '3 Years', '5 Years']

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)


# --- Helper Functions ---
def calculate_monthly_returns(series):
    monthly_series = series.resample('M').last()
    return monthly_series.pct_change()

@st.cache_data(ttl="1h", show_spinner="Loading portfolio allocation data...")
def read_portfolios_from_google_sheet(sheet_id):
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
        st.error(f"Error reading from Google Sheet. Error: {e}")
        return {}

@st.cache_data(ttl="6h")
def get_names_from_codes(scheme_codes_list):
    names = {}
    scheme_codes_list = [str(c) for c in scheme_codes_list if c]
    for code in scheme_codes_list:
        try:
            response = requests.get(f"https://api.mfapi.in/mf/{code}")
            if response.status_code == 200:
                scheme_name = response.json().get("meta", {}).get("scheme_name", f"Unknown: {code}")
                names[str(code)] = scheme_name
            else: names[str(code)] = f"Failed: {code}"
        except Exception as e:
            logging.warning(f"Error fetching scheme name for {code}: {e}")
            names[str(code)] = f"Error: {code}"
    return names


@st.cache_data(ttl="1h", show_spinner="Fetching all historical NAV data...")
def _fetch_full_nav_history(scheme_codes_tuple):
    all_nav_history = {}
    progress_bar = st.progress(0, text=f"Fetching NAVs...")
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
            else: logging.warning(f"Failed NAV fetch for {code}: HTTP {response.status_code}")
        except Exception as e:
            logging.warning(f"Exception NAV fetch for {code}: {e}")
        finally:
            progress_bar.progress((i + 1) / len(scheme_codes_tuple), text=f"Fetching NAVs...({i+1}/{len(scheme_codes_tuple)})")
    progress_bar.empty()
    return all_nav_history


def calculate_trailing_returns(series):
    returns = {}
    series = series.sort_index().dropna()
    if len(series) < 2: return pd.Series(dtype=float)
    end_date, end_value = series.index[-1], series.iloc[-1]
    special_periods = {'MTD': datetime(end_date.year, end_date.month, 1), 'YTD': pd.to_datetime(f'{end_date.year - 1}-12-31')}
    for period_name, start_date_target in special_periods.items():
        try:
            position = series.index.get_indexer([start_date_target], method='pad')[0]
            actual_start_date, start_value = series.index[position], series.iloc[position]
            if pd.notna(start_value) and start_value > 0 and actual_start_date < end_date:
                returns[period_name] = (end_value / start_value) - 1
        except (IndexError, KeyError): continue
    periods = {'1 Month': relativedelta(months=1), '3 Months': relativedelta(months=3), '6 Months': relativedelta(months=6), '1 Year': relativedelta(years=1), '3 Years': relativedelta(years=3), '5 Years': relativedelta(years=5)}
    for period_name, delta in periods.items():
        start_date_target = end_date - delta
        try:
            position = series.index.get_indexer([start_date_target], method='pad')[0]
            actual_start_date, start_value = series.index[position], series.iloc[position]
            if pd.notna(start_value) and start_value > 0 and actual_start_date < end_date:
                days_in_period = (end_date - actual_start_date).days
                if (delta.years is not None and delta.years >= 1) or (delta.months is not None and delta.months >= 12 and days_in_period > 200):
                     returns[period_name] = ((end_value / start_value) ** (365.25 / days_in_period)) - 1
                else: returns[period_name] = (end_value / start_value) - 1
        except (IndexError, KeyError): continue
    return pd.Series(returns)

def style_table(styler, format_str, na_rep, cmap, weight_col=None):
    cols_to_format = list(styler.data.columns)
    if weight_col and weight_col in cols_to_format:
        styler.format({weight_col: '{:,.2%}'})
        cols_to_format.remove(weight_col)
    gradient_cols = []
    numeric_cols = styler.data.select_dtypes(include=np.number).columns
    for col in cols_to_format:
        if col in numeric_cols and styler.data[col].notna().any():
            gradient_cols.append(col)
    styler.format(format_str, subset=cols_to_format, na_rep=na_rep)
    if gradient_cols:
        styler.background_gradient(cmap=cmap, subset=gradient_cols, axis=0)
    styler.map_index(lambda v: 'text-align: left;')
    return styler

@st.cache_data(show_spinner="Calculating portfolio performance...")
def perform_full_analysis(_all_portfolios_data_original, _all_navs_df, _start_date, _end_date, _initial_investment):
    start_ts, end_ts = pd.to_datetime(_start_date), pd.to_datetime(_end_date)
    all_portfolios_data = {}
    skipped_portfolios_date_range = []
    for name, df in _all_portfolios_data_original.items():
        df = df.loc[:, ~df.columns.duplicated()]
        filtered_allocations = df.loc[:, (df.columns >= start_ts) & (df.columns <= end_ts)]
        if filtered_allocations.shape[1] >= 2:
            all_portfolios_data[name] = filtered_allocations
        elif not filtered_allocations.empty:
            skipped_portfolios_date_range.append(name)
    if not all_portfolios_data: return {}, {}, skipped_portfolios_date_range, [], {}
    navs_df_filtered = _all_navs_df.loc[start_ts:end_ts]
    portfolio_results, skipped_portfolios_no_data, dropped_funds_info = {}, [], {}
    all_daily_returns = navs_df_filtered.pct_change()
    latest_date, earliest_date = navs_df_filtered.index.max(), navs_df_filtered.index.min()
    for name, allocations_original in all_portfolios_data.items():
        valid_codes = allocations_original.index.intersection(navs_df_filtered.columns)
        dropped_codes = allocations_original.index.difference(valid_codes).tolist()
        if dropped_codes: dropped_funds_info[name] = dropped_codes
        if valid_codes.empty:
            skipped_portfolios_no_data.append(name)
            continue
        allocations = allocations_original.loc[valid_codes].div(allocations_original.loc[valid_codes].sum(axis=0), axis=1).fillna(0)
        portfolio_start_date = allocations.columns.min()
        date_range = pd.date_range(start=portfolio_start_date, end=latest_date, freq='D')
        portfolio_fund_returns = all_daily_returns[allocations.index].reindex(date_range).fillna(0)
        daily_target_allocations = allocations.T.reindex(date_range, method='ffill')
        daily_value_index = pd.Series(index=date_range, dtype=float)
        daily_value_index.iloc[0] = _initial_investment
        holdings_value = pd.DataFrame(index=date_range, columns=allocations.index, dtype=float)
        holdings_value.iloc[0] = _initial_investment * daily_target_allocations.iloc[0]
        for t in range(1, len(date_range)):
            prev_date, current_date = date_range[t-1], date_range[t]
            grown_holdings = holdings_value.loc[prev_date] * (1 + portfolio_fund_returns.loc[current_date])
            if not daily_target_allocations.loc[current_date].equals(daily_target_allocations.loc[prev_date]):
                holdings_value.loc[current_date] = grown_holdings.sum() * daily_target_allocations.loc[current_date]
            else:
                holdings_value.loc[current_date] = grown_holdings
            daily_value_index.loc[current_date] = holdings_value.loc[current_date].sum()
        daily_value_index = daily_value_index.dropna()
        periodic_navs = navs_df_filtered.reindex(allocations.columns, method='ffill')
        portfolio_results[name] = {
            'allocations': allocations, 'daily_value_index': daily_value_index,
            'portfolio_trailing_returns': calculate_trailing_returns(daily_value_index),
            'periodic_fund_returns': periodic_navs.loc[:, allocations.index].pct_change().T,
            'periodic_portfolio_returns': (allocations.shift(1, axis=1).T * periodic_navs.loc[:, allocations.index].pct_change()).sum(axis=1, min_count=1),
            'benchmark_periodic_returns': periodic_navs[list(BENCHMARKS.values())].pct_change().rename(columns=dict(zip(BENCHMARKS.values(), BENCHMARKS.keys()))).T,
            'fund_trailing_returns': ((1 + portfolio_fund_returns).cumprod() * _initial_investment).apply(calculate_trailing_returns, axis=0).T,
        }
    benchmark_daily_indices = {}
    unified_date_range = pd.date_range(start=earliest_date, end=latest_date, freq='D')
    for b_name, b_code in BENCHMARKS.items():
        if b_code in all_daily_returns:
            b_index = (1 + all_daily_returns[b_code].reindex(unified_date_range).fillna(0)).cumprod() * _initial_investment
            benchmark_daily_indices[b_name] = b_index
    return portfolio_results, benchmark_daily_indices, skipped_portfolios_date_range, skipped_portfolios_no_data, dropped_funds_info

# --- Main App ---
st.title("🚀 Comprehensive Portfolio Performance Dashboard")
st.markdown("Analyse portfolio performance using periodic returns from a live data source and up-to-date trailing returns from market data.")
try:
    google_sheet_id = st.secrets["GOOGLE_SHEET_ID"]
    all_portfolios_data_original = read_portfolios_from_google_sheet(google_sheet_id)
except KeyError:
    st.error("`GOOGLE_SHEET_ID` not found in Streamlit secrets.")
    st.stop()
if not all_portfolios_data_original:
    st.warning("No portfolio data loaded. Check Google Sheet format/sharing.")
    st.stop()
all_fund_codes = set(code for p in all_portfolios_data_original.values() for code in p.index)
all_scheme_codes = tuple(sorted(list(all_fund_codes | set(BENCHMARKS.values()))))
full_nav_history = _fetch_full_nav_history(all_scheme_codes)
if not full_nav_history:
    st.error("Could not fetch NAV data for any funds. Check scheme codes.")
    st.stop()
all_navs_df = pd.DataFrame(full_nav_history).ffill().bfill()

if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
def mark_rerun_required():
    st.session_state.analysis_run = False
with st.sidebar:
    st.header("⚙️ Controls")
    initial_investment = st.number_input("1. Initial Investment", min_value=1.0, value=10000.0, step=1000.0, on_change=mark_rerun_required)
    api_min_date, api_max_date = all_navs_df.index.min().date(), all_navs_df.index.max().date()
    st.markdown("---")
    st.header("2. Set Date Range")
    start_date = st.date_input("Analysis Start Date", value=api_min_date, min_value=api_min_date, max_value=api_max_date, on_change=mark_rerun_required)
    
    ### --- CRASH FIX: Ensure default end_date is not after max_value --- ###
    safe_default_end_date = min(date.today(), api_max_date)
    end_date = st.date_input("Analysis End Date", value=safe_default_end_date, min_value=api_min_date, max_value=api_max_date, on_change=mark_rerun_required)
    
    st.markdown("---")
    run_button = st.button("📊 Run Analysis", type="primary", use_container_width=True)

should_run = run_button or not st.session_state.analysis_run
if should_run:
    st.session_state.analysis_run = True
    if start_date > end_date:
        st.error("Error: End date must be on or after start date.")
        st.stop()
    
    portfolio_results, benchmark_daily_indices, skipped_date, skipped_data, dropped = perform_full_analysis(
        all_portfolios_data_original, all_navs_df, start_date, end_date, initial_investment
    )

    if skipped_date: st.warning(f"Skipped (fewer than 2 rebalance dates in range): **{', '.join(skipped_date)}**")
    if skipped_data: st.warning(f"Skipped (no valid fund NAVs found): **{', '.join(skipped_data)}**")
    if dropped:
        st.warning("Some funds were excluded from analysis (weights renormalized):")
        all_dropped = tuple(set(code for codes in dropped.values() for code in codes))
        names_map = get_names_from_codes(all_dropped)
        for name, codes in dropped.items():
            names = [f"{names_map.get(c, c)}" for c in codes]
            st.markdown(f"- **{name}**: Dropped {', '.join(names)}")
    if not portfolio_results:
        st.error("No portfolios to display after filtering.")
        st.stop()
    
    all_portfolio_codes = tuple(set(code for res in portfolio_results.values() for code in res['allocations'].index))
    st.session_state.names_map = get_names_from_codes(all_portfolio_codes)
    st.session_state.portfolio_results = portfolio_results
    st.session_state.benchmark_daily_indices = benchmark_daily_indices
    st.session_state.excel_cmap = LinearSegmentedColormap.from_list("excel_like", ["#f8696b", "#ffeb84", "#63be7b"])

if 'portfolio_results' in st.session_state and st.session_state.portfolio_results:
    portfolio_results = st.session_state.portfolio_results
    benchmark_daily_indices = st.session_state.benchmark_daily_indices
    names_map = st.session_state.names_map
    excel_cmap = st.session_state.excel_cmap
    tab_names = ["📈 Comparison"] + list(portfolio_results.keys())
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.header("Overall Portfolio Comparison")
        st.subheader("Trailing Returns Comparison")
        benchmark_trailing = pd.DataFrame({name: calculate_trailing_returns(series) for name, series in benchmark_daily_indices.items()}).T
        portfolio_trailing = pd.DataFrame({n: r['portfolio_trailing_returns'] for n, r in portfolio_results.items()}).T
        comparison_trailing_df = pd.concat([portfolio_trailing, benchmark_trailing])
        final_cols = [c for c in TRAILING_COLS_ORDER if c in comparison_trailing_df.columns]
        st.dataframe(style_table(comparison_trailing_df[final_cols].style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)

        st.subheader("Month-on-Month Returns")
        portfolio_mom = pd.DataFrame({name: calculate_monthly_returns(res['daily_value_index']) for name, res in portfolio_results.items()}).T
        portfolio_mom.columns = portfolio_mom.columns.strftime('%b-%Y')
        st.markdown("##### **Portfolios**")
        st.dataframe(style_table(portfolio_mom.style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)
        benchmark_mom = pd.DataFrame({name: calculate_monthly_returns(series) for name, series in benchmark_daily_indices.items()}).T
        benchmark_mom.columns = benchmark_mom.columns.strftime('%b-%Y')
        st.markdown("##### **Benchmarks**")
        st.dataframe(style_table(benchmark_mom.style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)

        st.subheader("Portfolio Value Growth Comparison")
        growth_df = pd.concat({n: r['daily_value_index'] for n, r in portfolio_results.items()}, axis=1)
        st.line_chart(growth_df)

    for i, (name, results) in enumerate(portfolio_results.items()):
        with tabs[i+1]:
            st.header(f"Performance Analysis for: {name}")
            st.subheader("📈 Portfolio Growth vs Benchmarks")
            start_val = results['daily_value_index'].iloc[0]
            start_date_ts = results['daily_value_index'].index.min()
            norm_bench = {b_name: (b_idx.loc[start_date_ts:] / b_idx.loc[start_date_ts]) * start_val for b_name, b_idx in benchmark_daily_indices.items() if not b_idx.loc[start_date_ts:].empty}
            combined_growth = pd.concat([results['daily_value_index'], pd.DataFrame(norm_bench)], axis=1)
            combined_growth.columns.values[0] = name
            st.line_chart(combined_growth)
            
            st.markdown("---")
            st.subheader("✅ Performance (Trailing Returns)")
            st.markdown("##### **Individual Funds**")
            fund_trailing = results['fund_trailing_returns'].copy()
            fund_trailing['Weight'] = results['allocations'].iloc[:, -1]
            fund_trailing.index = fund_trailing.index.map(names_map)
            final_cols = ['Weight'] + [c for c in TRAILING_COLS_ORDER if c in fund_trailing.columns]
            st.dataframe(style_table(fund_trailing[final_cols].style, '{:.2%}', 'N/A', excel_cmap, 'Weight'), use_container_width=True)
            
            st.markdown("##### **Portfolio vs. Benchmarks**")
            port_trailing = results['portfolio_trailing_returns']
            port_trailing.name = name
            bench_trailing = pd.DataFrame({b_name: calculate_trailing_returns(b_idx) for b_name, b_idx in norm_bench.items()}).T
            combined_trailing = pd.concat([port_trailing.to_frame().T, bench_trailing])
            final_cols = [c for c in TRAILING_COLS_ORDER if c in combined_trailing.columns]
            st.dataframe(style_table(combined_trailing[final_cols].style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)
            st.markdown("---")
            st.subheader("✅ Performance Between Rebalancing Dates (Periodic Returns)")
            st.markdown("##### **Individual Funds (Periodic)**")
            df_periodic = results['periodic_fund_returns'].copy()
            df_periodic['Weight'] = results['allocations'].iloc[:, -1]
            df_periodic.index = df_periodic.index.map(names_map)
            df_periodic.columns = [c.strftime('%d-%b-%Y') if isinstance(c, pd.Timestamp) else c for c in df_periodic.columns]
            if 'Weight' in df_periodic.columns:
                cols = df_periodic.columns.tolist()
                cols.insert(0, cols.pop(cols.index('Weight')))
                df_periodic = df_periodic[cols]
            df_periodic = df_periodic.loc[:, ~df_periodic.columns.duplicated()]
            st.dataframe(style_table(df_periodic.style, '{:.2f}%', 'None', excel_cmap, 'Weight'), use_container_width=True)
            st.markdown("##### **Portfolio vs. Benchmarks (Periodic)**")
            port_periodic = results['periodic_portfolio_returns'].mul(100)
            port_periodic.name = f"📊 {name} Portfolio"
            bench_periodic = results['benchmark_periodic_returns'].mul(100)
            combined_periodic = pd.concat([port_periodic.to_frame().T, bench_periodic])
            combined_periodic.columns = [c.strftime('%d-%b-%Y') for c in combined_periodic.columns]
            combined_periodic = combined_periodic.loc[:, ~combined_periodic.columns.duplicated()]
            st.dataframe(style_table(combined_periodic.style, '{:.2f}%', 'None', excel_cmap), use_container_width=True)