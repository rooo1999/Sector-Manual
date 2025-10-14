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

# --- Global Constants & Logging ---
BENCHMARKS = {
    "Nifty 50 TRI": "147794",
    "Nifty 500 TRI": "147625",
    "Smallcap 250 TRI": "147623",
    "Midcap 150 TRI": "147622",
    "Sensex TRI": "119065"
}
TRAILING_COLS_ORDER = ['MTD', 'YTD', '1 Month', '3 Months', '6 Months', '1 Year', '3 Years', '5 Years']
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)


# --- Helper Functions (No changes needed in this section) ---
def calculate_monthly_returns(series):
    return series.resample('M').last().pct_change()

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
                if np.isclose(df[col].sum(), 100.0, atol=0.1): df[col] = df[col] / 100.0
            parsed_columns = {col: pd.to_datetime(col, dayfirst=False, errors='coerce') for col in df.columns}
            df = df.rename(columns=parsed_columns)
            df = df[[c for c in df.columns if isinstance(c, pd.Timestamp)]].sort_index(axis=1)
            if not df.empty: cleaned_portfolios[sheet_name] = df
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
                names[str(code)] = response.json().get("meta", {}).get("scheme_name", f"Unknown: {code}")
            else: names[str(code)] = f"Failed: {code}"
        except Exception: names[str(code)] = f"Error: {code}"
    return names

@st.cache_data(ttl="1h", show_spinner="Fetching all historical NAV data...")
def _fetch_full_nav_history(scheme_codes_tuple):
    all_nav_history = {}
    progress_bar = st.progress(0, text=f"Fetching NAVs...")
    for i, code in enumerate(scheme_codes_tuple):
        try:
            url = f"https://api.mfapi.in/mf/{code}"
            response = requests.get(url)
            if response.status_code == 200 and response.json().get("data"):
                df = pd.DataFrame(response.json()["data"])
                df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
                df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
                all_nav_history[code] = df.dropna().set_index('date').sort_index()['nav']
        except Exception: pass
        finally: progress_bar.progress((i + 1) / len(scheme_codes_tuple), text=f"Fetching NAVs...({i+1}/{len(scheme_codes_tuple)})")
    progress_bar.empty()
    return all_nav_history

def calculate_trailing_returns(series):
    returns = {}
    series = series.sort_index().dropna()
    if len(series) < 2: return pd.Series(dtype=float)
    end_date, end_value = series.index[-1], series.iloc[-1]
    special_periods = {'MTD': datetime(end_date.year, end_date.month, 1), 'YTD': pd.to_datetime(f'{end_date.year - 1}-12-31')}
    for name, target_date in special_periods.items():
        try:
            pos = series.index.get_indexer([target_date], method='pad')[0]
            if pos != -1 and series.index[pos] < end_date:
                returns[name] = (end_value / series.iloc[pos]) - 1
        except (IndexError, KeyError): continue
    periods = {'1 Month': relativedelta(months=1), '3 Months': relativedelta(months=3), '6 Months': relativedelta(months=6), '1 Year': relativedelta(years=1), '3 Years': relativedelta(years=3), '5 Years': relativedelta(years=5)}
    for name, delta in periods.items():
        try:
            pos = series.index.get_indexer([end_date - delta], method='pad')[0]
            if pos != -1 and series.index[pos] < end_date:
                start_val, start_date = series.iloc[pos], series.index[pos]
                days = (end_date - start_date).days
                if 'Year' in name and days > 200:
                    returns[name] = ((end_value / start_val) ** (365.25 / days)) - 1
                else:
                    returns[name] = (end_value / start_val) - 1
        except (IndexError, KeyError): continue
    return pd.Series(returns)

def style_table(styler, format_str, na_rep, cmap, weight_col=None):
    cols_to_format = list(styler.data.columns)
    if weight_col and weight_col in cols_to_format:
        styler.format({weight_col: '{:,.2%}'})
        cols_to_format.remove(weight_col)
    gradient_cols = [c for c in cols_to_format if c in styler.data.select_dtypes(include=np.number).columns and styler.data[c].notna().any()]
    styler.format(format_str, subset=cols_to_format, na_rep=na_rep)
    if gradient_cols: styler.background_gradient(cmap=cmap, subset=gradient_cols, axis=0)
    styler.map_index(lambda v: 'text-align: left;')
    return styler

@st.cache_data(show_spinner="Calculating portfolio performance...")
def perform_full_analysis(_all_portfolios_data_original, _all_navs_df, _start_date, _end_date, _initial_investment):
    start_ts, end_ts = pd.to_datetime(_start_date), pd.to_datetime(_end_date)
    all_portfolios_data = {}
    skipped_portfolios_date_range = []
    for name, df in _all_portfolios_data_original.items():
        df = df.loc[:, ~df.columns.duplicated()]
        filtered = df.loc[:, (df.columns >= start_ts) & (df.columns <= end_ts)]
        if filtered.shape[1] >= 2: all_portfolios_data[name] = filtered
        elif not filtered.empty: skipped_portfolios_date_range.append(name)
    if not all_portfolios_data: return {}, {}, skipped_portfolios_date_range, [], {}
    navs_df_filtered = _all_navs_df.loc[start_ts:end_ts]
    results, skipped_no_data, dropped_funds = {}, [], {}
    all_daily_returns = navs_df_filtered.pct_change()
    latest, earliest = navs_df_filtered.index.max(), navs_df_filtered.index.min()
    for name, alloc_orig in all_portfolios_data.items():
        valid_codes = alloc_orig.index.intersection(navs_df_filtered.columns)
        if valid_codes.empty:
            skipped_no_data.append(name)
            continue
        dropped = alloc_orig.index.difference(valid_codes).tolist()
        if dropped: dropped_funds[name] = dropped
        alloc = alloc_orig.loc[valid_codes].div(alloc_orig.loc[valid_codes].sum(axis=0), axis=1).fillna(0)
        start = alloc.columns.min()
        date_range = pd.date_range(start=start, end=latest, freq='D')
        fund_returns = all_daily_returns[alloc.index].reindex(date_range).fillna(0)
        target_allocs = alloc.T.reindex(date_range, method='ffill')
        holdings = pd.DataFrame(0.0, index=date_range, columns=alloc.index)
        holdings.iloc[0] = _initial_investment * target_allocs.iloc[0]
        for t in range(1, len(date_range)):
            grown = holdings.iloc[t-1] * (1 + fund_returns.iloc[t])
            if not target_allocs.iloc[t].equals(target_allocs.iloc[t-1]):
                holdings.iloc[t] = grown.sum() * target_allocs.iloc[t]
            else:
                holdings.iloc[t] = grown
        daily_value = holdings.sum(axis=1).replace(0, np.nan).dropna()
        periodic_navs = navs_df_filtered.reindex(alloc.columns, method='ffill')
        results[name] = {
            'allocations': alloc, 'daily_value_index': daily_value,
            'portfolio_trailing_returns': calculate_trailing_returns(daily_value),
            'periodic_fund_returns': periodic_navs.loc[:, alloc.index].pct_change().T,
            'periodic_portfolio_returns': (alloc.shift(1, axis=1).T * periodic_navs.loc[:, alloc.index].pct_change()).sum(axis=1, min_count=1),
            'benchmark_periodic_returns': periodic_navs[list(BENCHMARKS.values())].pct_change().rename(columns=dict(zip(BENCHMARKS.values(), BENCHMARKS.keys()))).T,
            'fund_trailing_returns': ((1 + fund_returns).cumprod() * _initial_investment).apply(calculate_trailing_returns, axis=0).T,
        }
    benchmark_indices = {}
    date_range_all = pd.date_range(start=earliest, end=latest, freq='D')
    for b_name, b_code in BENCHMARKS.items():
        if b_code in all_daily_returns:
            benchmark_indices[b_name] = (1 + all_daily_returns[b_code].reindex(date_range_all).fillna(0)).cumprod() * _initial_investment
    return results, benchmark_indices, skipped_portfolios_date_range, skipped_no_data, dropped_funds

# --- App Layout & Data Loading ---
st.title("🚀 Comprehensive Portfolio Performance Dashboard")
try:
    google_sheet_id = st.secrets["GOOGLE_SHEET_ID"]
    all_portfolios_data_original = read_portfolios_from_google_sheet(google_sheet_id)
except KeyError: st.error("`GOOGLE_SHEET_ID` not found in Streamlit secrets."); st.stop()
if not all_portfolios_data_original: st.warning("No portfolio data loaded."); st.stop()
all_scheme_codes = tuple(sorted(list(set(c for p in all_portfolios_data_original.values() for c in p.index) | set(BENCHMARKS.values()))))
full_nav_history = _fetch_full_nav_history(all_scheme_codes)
if not full_nav_history: st.error("Could not fetch NAV data for any funds."); st.stop()
all_navs_df = pd.DataFrame(full_nav_history).ffill().bfill()

# --- Session State & Sidebar Controls ---
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
def mark_rerun_required(): st.session_state.analysis_run = False
with st.sidebar:
    st.header("⚙️ Controls")
    initial_investment = st.number_input("1. Initial Investment", 1.0, value=10000.0, step=1000.0, on_change=mark_rerun_required)
    api_min, api_max = all_navs_df.index.min().date(), all_navs_df.index.max().date()
    st.markdown("---")
    st.header("2. Set Date Range")
    start_date = st.date_input("Start Date", api_min, api_min, api_max, on_change=mark_rerun_required)
    safe_end_date = min(date.today(), api_max)
    end_date = st.date_input("End Date", safe_end_date, api_min, api_max, on_change=mark_rerun_required)
    st.markdown("---")
    run_button = st.button("📊 Run Analysis", type="primary", use_container_width=True)

# --- Main Calculation Block ---
should_run = run_button or not st.session_state.analysis_run
if should_run:
    st.session_state.clear() # Clear old results
    st.session_state.analysis_run = True
    if start_date > end_date:
        st.error("Error: End date must be on or after start date.")
    else:
        results, bench_indices, skipped_d, skipped_n, dropped = perform_full_analysis(
            all_portfolios_data_original, all_navs_df, start_date, end_date, initial_investment
        )
        if skipped_d: st.warning(f"Skipped (date range): **{', '.join(skipped_d)}**")
        if skipped_n: st.warning(f"Skipped (no NAVs): **{', '.join(skipped_n)}**")
        if dropped:
            st.warning("Excluded funds:")
            all_dropped_codes = tuple(set(c for codes in dropped.values() for c in codes))
            names_map_dropped = get_names_from_codes(all_dropped_codes)
            for name, codes in dropped.items():
                names = [f"{names_map_dropped.get(c, c)}" for c in codes]
                st.markdown(f"- **{name}**: {', '.join(names)}")
        
        if results:
            ### --- PERFORMANCE FIX: PRE-CALCULATE EVERYTHING HERE --- ###
            st.session_state.portfolio_results = results
            st.session_state.benchmark_daily_indices = bench_indices
            all_codes = tuple(set(c for res in results.values() for c in res['allocations'].index))
            st.session_state.names_map = get_names_from_codes(all_codes)
            
            # Pre-calculate Comparison Tab data
            st.session_state.comparison_trailing = pd.concat([
                pd.DataFrame({n: r['portfolio_trailing_returns'] for n, r in results.items()}).T,
                pd.DataFrame({n: calculate_trailing_returns(s) for n, s in bench_indices.items()}).T
            ])
            st.session_state.portfolio_mom = pd.DataFrame({n: calculate_monthly_returns(r['daily_value_index']) for n, r in results.items()}).T
            st.session_state.benchmark_mom = pd.DataFrame({n: calculate_monthly_returns(s) for n, s in bench_indices.items()}).T
            st.session_state.growth_df = pd.concat({n: r['daily_value_index'] for n, r in results.items()}, axis=1)

# --- UI Rendering Block (Purely for Display) ---
if 'portfolio_results' in st.session_state:
    excel_cmap = LinearSegmentedColormap.from_list("excel_like", ["#f8696b", "#ffeb84", "#63be7b"])
    names_map = st.session_state.names_map
    portfolio_results = st.session_state.portfolio_results
    
    tab_names = ["📈 Comparison"] + list(portfolio_results.keys())
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.header("Overall Portfolio Comparison")
        st.subheader("Trailing Returns Comparison")
        df = st.session_state.comparison_trailing
        final_cols = [c for c in TRAILING_COLS_ORDER if c in df.columns]
        st.dataframe(style_table(df[final_cols].style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)

        st.subheader("Month-on-Month Returns")
        st.markdown("##### **Portfolios**")
        df_p = st.session_state.portfolio_mom
        df_p.columns = df_p.columns.strftime('%b-%Y')
        st.dataframe(style_table(df_p.style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)
        st.markdown("##### **Benchmarks**")
        df_b = st.session_state.benchmark_mom
        df_b.columns = df_b.columns.strftime('%b-%Y')
        st.dataframe(style_table(df_b.style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)
        
        st.subheader("Portfolio Value Growth Comparison")
        st.line_chart(st.session_state.growth_df)

    for i, (name, results) in enumerate(portfolio_results.items()):
        with tabs[i+1]:
            st.header(f"Performance Analysis for: {name}")
            
            st.subheader("📈 Portfolio Growth vs Benchmarks")
            start_val = results['daily_value_index'].iloc[0]
            start_date_ts = results['daily_value_index'].index.min()
            norm_bench = {
                b_name: (b_idx.loc[start_date_ts:] / b_idx.loc[start_date_ts]) * start_val
                for b_name, b_idx in st.session_state.benchmark_daily_indices.items() if not b_idx.loc[start_date_ts:].empty
            }
            combined_growth = pd.concat([results['daily_value_index'], pd.DataFrame(norm_bench)], axis=1)
            combined_growth.columns.values[0] = name
            st.line_chart(combined_growth)
            
            st.markdown("---")
            st.subheader("✅ Performance (Trailing Returns)")
            st.markdown("##### **Individual Funds**")
            fund_trailing = results['fund_trailing_returns'].copy()
            fund_trailing['Weight'] = results['allocations'].iloc[:, -1]
            fund_trailing.index = fund_trailing.index.map(names_map)
            final_cols_ft = ['Weight'] + [c for c in TRAILING_COLS_ORDER if c in fund_trailing.columns]
            st.dataframe(style_table(fund_trailing[final_cols_ft].style, '{:.2%}', 'N/A', excel_cmap, 'Weight'), use_container_width=True)
            
            st.markdown("##### **Portfolio vs. Benchmarks**")
            port_trailing = results['portfolio_trailing_returns']
            port_trailing.name = name
            bench_trailing = pd.DataFrame({b_name: calculate_trailing_returns(b_idx) for b_name, b_idx in norm_bench.items()}).T
            combined_trailing = pd.concat([port_trailing.to_frame().T, bench_trailing])
            final_cols_ct = [c for c in TRAILING_COLS_ORDER if c in combined_trailing.columns]
            st.dataframe(style_table(combined_trailing[final_cols_ct].style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)
            
            st.markdown("---")
            st.subheader("✅ Performance Between Rebalancing Dates (Periodic Returns)")
            st.markdown("##### **Individual Funds (Periodic)**")
            df_periodic = results['periodic_fund_returns'].copy()
            df_periodic['Weight'] = results['allocations'].iloc[:, -1]
            df_periodic.index = df_periodic.index.map(names_map)
            df_periodic.columns = [c.strftime('%d-%b-%Y') if isinstance(c, pd.Timestamp) else c for c in df_periodic.columns]
            cols = df_periodic.columns.tolist()
            if 'Weight' in cols:
                cols.insert(0, cols.pop(cols.index('Weight')))
                df_periodic = df_periodic[cols]
            df_periodic = df_periodic.loc[:, ~df_periodic.columns.duplicated()]
            st.dataframe(style_table(df_periodic.style, '{:.2f}%', 'None', excel_cmap, 'Weight'), use_container_width=True)

            st.markdown("##### **Portfolio vs. Benchmarks (Periodic)**")
            port_periodic = results['periodic_portfolio_returns'].mul(100)
            port_periodic.name = f"📊 {name} Portfolio"
            bench_periodic = results['benchmark_periodic_returns'].mul(100)
            combined_periodic = pd.concat([port_periodic.to_frame().T, bench_periodic])
            combined_periodic.columns = [c.strftime('%d-%b-%Y') if isinstance(c, pd.Timestamp) else c for c in combined_periodic.columns]
            combined_periodic = combined_periodic.loc[:, ~combined_periodic.columns.duplicated()]
            st.dataframe(style_table(combined_periodic.style, '{:.2f}%', 'None', excel_cmap), use_container_width=True)