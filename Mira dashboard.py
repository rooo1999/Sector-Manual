import streamlit as st
import pandas as pd
import numpy as np
import requests
import logging
import sys
from urllib.parse import quote
from datetime import datetime
from dateutil.relativedelta import relativedelta
from matplotlib.colors import LinearSegmentedColormap

try:
    import yfinance as yf
except ImportError:
    yf = None

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="MIRA Money — Fund & Market Dashboard",
    page_icon="📊",
    layout="wide"
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================

# Same indices as the reference dashboard
INDICES = {
    "Nifty 50 TRI": "147794",
    "Nifty 500 TRI": "147625",
    "Smallcap 250 TRI": "147623",
    "Midcap 150 TRI": "147622",
    "Sensex TRI": "119065"
}

TRAILING_COLS_ORDER = ['MTD', 'YTD', '1 Month', '3 Months', '6 Months', '1 Year', '3 Years', '5 Years']

# Macro factors: label -> (source, ticker/column, format string for level values)
# source is either "yfinance" (pulled live) or "sheet" (pulled from the Macro tab of the Google Sheet,
# since Nifty PE and India 10Y G-Sec have no reliable free public API)
MACRO_FACTORS = [
    {"label": "Nifty 50 PE",            "source": "sheet",     "key": "Nifty50 PE",        "fmt": "{:,.2f}"},
    {"label": "India 10Y G-Sec (%)",    "source": "sheet",     "key": "India 10Y Gsec",    "fmt": "{:,.2f}"},
    {"label": "US 10Y Treasury (%)",    "source": "yfinance",  "key": "^TNX",              "fmt": "{:,.2f}", "divide": 10.0},
    {"label": "USD/INR",                "source": "yfinance",  "key": "USDINR=X",          "fmt": "{:,.2f}"},
    {"label": "Gold (GOLDBEES, ₹)",     "source": "yfinance",  "key": "GOLDBEES.NS",       "fmt": "{:,.2f}"},
    {"label": "Silver (SILVERBEES, ₹)", "source": "yfinance",  "key": "SILVERBEES.NS",     "fmt": "{:,.2f}"},
]

NIFTY_PRICE_TICKER = "^NSEI"  # used for the 50-day / 200-day moving averages

excel_cmap = LinearSegmentedColormap.from_list("excel_like", ["#f8696b", "#ffeb84", "#63be7b"])

if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False

# ============================================================================
# DATA LOADERS — GOOGLE SHEET (fund lists + manual macro data)
# ============================================================================

def _sheet_csv_url(sheet_id, tab_name):
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(tab_name)}"


@st.cache_data(ttl="1h", show_spinner="Loading fund list from Google Sheet...")
def read_fund_list_sheet(sheet_id, tab_name):
    """
    Reads a tab that lists funds to track.
    Expected columns: 'Scheme Code', 'Category'  (Category is optional — defaults to the tab name)
    """
    url = _sheet_csv_url(sheet_id, tab_name)
    try:
        df = pd.read_csv(url, dtype=str)
    except Exception as e:
        st.error(f"Could not read tab '{tab_name}' from the Google Sheet. Error: {e}")
        return pd.DataFrame(columns=['Scheme Code', 'Category'])

    df.columns = [str(c).strip() for c in df.columns]
    if 'Scheme Code' not in df.columns:
        # fall back to first column
        df = df.rename(columns={df.columns[0]: 'Scheme Code'})
    df['Scheme Code'] = df['Scheme Code'].astype(str).str.strip()
    df = df[df['Scheme Code'].notna() & (df['Scheme Code'] != '') & (df['Scheme Code'].str.lower() != 'nan')]

    if 'Category' not in df.columns:
        df['Category'] = tab_name
    df['Category'] = df['Category'].fillna(tab_name).replace('', tab_name)

    return df[['Scheme Code', 'Category']].drop_duplicates(subset='Scheme Code')


@st.cache_data(ttl="1h", show_spinner="Loading macro data from Google Sheet...")
def read_macro_sheet(sheet_id, tab_name="Macro"):
    """
    Reads the manually-maintained macro tab.
    Expected columns: 'Date', 'Nifty50 PE', 'India 10Y Gsec'  (add more columns anytime — they're matched by name)
    """
    url = _sheet_csv_url(sheet_id, tab_name)
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.warning(f"Could not read the '{tab_name}' tab from the Google Sheet. Error: {e}")
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


# ============================================================================
# DATA LOADERS — MFAPI (NAV history + scheme names)
# ============================================================================

@st.cache_data(ttl="6h")
def get_names_from_codes(scheme_codes_list):
    names = {}
    for code in scheme_codes_list:
        try:
            response = requests.get(f"https://api.mfapi.in/mf/{code}", timeout=10)
            if response.status_code == 200:
                names[str(code)] = response.json().get("meta", {}).get("scheme_name", f"Unknown: {code}")
            else:
                names[str(code)] = f"Unknown: {code}"
        except Exception as e:
            logging.warning(f"Error fetching scheme name for {code}: {e}")
            names[str(code)] = f"Unknown: {code}"
    return names


@st.cache_data(ttl="1h", show_spinner="Fetching NAV history...")
def fetch_nav_history(scheme_codes_tuple):
    all_nav_history = {}
    progress_bar = st.progress(0, text="Fetching NAVs...")
    for i, code in enumerate(scheme_codes_tuple):
        try:
            response = requests.get(f"https://api.mfapi.in/mf/{code}", timeout=15)
            if response.status_code == 200:
                nav_data = response.json().get("data", [])
                if nav_data:
                    df = pd.DataFrame(nav_data)
                    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
                    df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
                    df = df.dropna().set_index('date').sort_index()
                    all_nav_history[code] = df['nav']
            else:
                logging.warning(f"Failed to fetch NAV history for {code}: HTTP {response.status_code}")
        except Exception as e:
            logging.warning(f"Exception fetching NAV history for {code}: {e}")
        finally:
            progress_bar.progress((i + 1) / len(scheme_codes_tuple), text=f"Fetching NAVs... ({i+1}/{len(scheme_codes_tuple)})")
    progress_bar.empty()
    return all_nav_history


# ============================================================================
# DATA LOADERS — YFINANCE (macro market data)
# ============================================================================

@st.cache_data(ttl="1h", show_spinner="Fetching macro market data...")
def fetch_yf_series(ticker, start="2015-01-01"):
    if yf is None:
        return pd.Series(dtype=float)
    try:
        data = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if data.empty:
            return pd.Series(dtype=float)
        s = data['Close']
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return s.dropna()
    except Exception as e:
        logging.warning(f"Error fetching {ticker} from yfinance: {e}")
        return pd.Series(dtype=float)


# ============================================================================
# CALCULATIONS
# ============================================================================

def calculate_trailing_returns(series):
    """Trailing returns using relativedelta and the financially-correct 'pad' method."""
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
    """Applies consistent styling (Excel-like red/yellow/green heatmap) to a Styler."""
    cols = list(styler.data.columns)
    if weight_col and weight_col in cols:
        styler.format({weight_col: '{:,.2%}'})
        cols.remove(weight_col)
    styler.format(format_str, subset=cols, na_rep=na_rep)
    styler.background_gradient(cmap=cmap, subset=cols, axis=0)
    styler.set_properties(**{'text-align': 'left'}, subset=pd.IndexSlice[:, :])
    return styler


def get_value_at_lag(series, end_date, delta):
    """Last available value on/before (end_date - delta), using the 'pad' method."""
    series = series.sort_index().dropna()
    if series.empty:
        return None, None
    target = end_date - delta
    try:
        pos = series.index.get_indexer([target], method='pad')[0]
        if pos == -1:
            return None, None
        return series.index[pos], series.iloc[pos]
    except Exception:
        return None, None


def build_macro_row(label, series):
    """Builds one row: Current, 1M/3M/6M/1Y Ago levels + % change vs current."""
    series = series.dropna().sort_index()
    row = {"Macro Factor": label}
    if series.empty:
        row["Current"] = np.nan
        for p in ["1M", "3M", "6M", "1Y"]:
            row[f"{p} Ago"] = np.nan
            row[f"{p} % Chg"] = np.nan
        return row

    end_date = series.index[-1]
    current = series.iloc[-1]
    row["Current"] = current

    periods = {
        "1M": relativedelta(months=1), "3M": relativedelta(months=3),
        "6M": relativedelta(months=6), "1Y": relativedelta(years=1)
    }
    for p_label, delta in periods.items():
        _, val = get_value_at_lag(series, end_date, delta)
        row[f"{p_label} Ago"] = val
        if val is not None and pd.notna(val) and val != 0:
            row[f"{p_label} % Chg"] = (current - val) / val
        else:
            row[f"{p_label} % Chg"] = np.nan
    return row


def style_macro_table(df):
    """Level columns get plain numeric formatting; % change columns get the heatmap."""
    df = df.set_index("Macro Factor")
    pct_cols = [c for c in df.columns if "% Chg" in c]
    val_cols = [c for c in df.columns if c not in pct_cols]

    fmt = {c: '{:,.2f}' for c in val_cols}
    fmt.update({c: '{:+.2%}' for c in pct_cols})

    styler = df.style.format(fmt, na_rep='N/A')
    styler.background_gradient(cmap=excel_cmap, subset=pct_cols, axis=0)
    styler.set_properties(**{'text-align': 'left'}, subset=pd.IndexSlice[:, :])
    return styler


def display_fund_group(fund_df, navs_df, fund_names_map, weights=None):
    """Renders one trailing-returns table per Category, in the same style as the reference dashboard."""
    if fund_df.empty:
        st.info("No funds listed on this tab of the Google Sheet yet.")
        return

    for category, group in fund_df.groupby('Category'):
        st.markdown(f"##### {category}")
        codes = group['Scheme Code'].tolist()
        trailing = {}
        for code in codes:
            if code in navs_df.columns:
                trailing[code] = calculate_trailing_returns(navs_df[code])
        if not trailing:
            st.info(f"No NAV data available yet for the funds in **{category}**.")
            continue

        df = pd.DataFrame(trailing).T
        df.index = df.index.map(lambda c: fund_names_map.get(c, c))

        weight_col = None
        if weights is not None:
            df['Weight'] = [weights.get(c, np.nan) for c in codes if c in trailing]
            weight_col = 'Weight'

        final_cols = ([weight_col] if weight_col else []) + [c for c in TRAILING_COLS_ORDER if c in df.columns]
        st.dataframe(style_table(df[final_cols].style, '{:.2%}', 'N/A', excel_cmap, weight_col), use_container_width=True)


# ============================================================================
# SIDEBAR
# ============================================================================
st.title("📊 MIRA Money — Fund & Market Dashboard")
st.caption("Live trailing returns for tracked funds, indices and watchlist, plus key macro indicators.")

with st.sidebar:
    st.header("⚙️ Controls")
    try:
        default_sheet_id = st.secrets.get("GOOGLE_SHEET_ID", "")
    except Exception:
        default_sheet_id = ""
    google_sheet_id = st.text_input("Google Sheet ID", value=default_sheet_id,
                                     help="The long ID in your sheet's URL. The sheet must be shared as 'Anyone with the link can view'.")

    st.markdown("---")
    st.markdown("**Expected tabs in the sheet:**")
    st.markdown(
        "- `Funds` — columns: `Scheme Code`, `Category`\n"
        "- `Watchlist` — columns: `Scheme Code`, `Category`\n"
        "- `Macro` — columns: `Date`, `Nifty50 PE`, `India 10Y Gsec`"
    )
    funds_tab = st.text_input("Funds tab name", value="Funds")
    watchlist_tab = st.text_input("Watchlist tab name", value="Watchlist")
    macro_tab = st.text_input("Macro tab name", value="Macro")

    st.markdown("---")
    run_button = st.button("🔄 Refresh Dashboard", type="primary", use_container_width=True)
    if run_button:
        st.cache_data.clear()

if not google_sheet_id:
    st.warning("👈 Enter your Google Sheet ID in the sidebar (or add `GOOGLE_SHEET_ID` to `.streamlit/secrets.toml`) to get started.")
    st.stop()

# ============================================================================
# LOAD FUND LISTS
# ============================================================================
funds_df = read_fund_list_sheet(google_sheet_id, funds_tab)
watchlist_df = read_fund_list_sheet(google_sheet_id, watchlist_tab)

all_fund_codes = set(funds_df['Scheme Code']) | set(watchlist_df['Scheme Code'])
all_scheme_codes = tuple(sorted(all_fund_codes | set(INDICES.values())))

if not all_scheme_codes:
    st.error("No scheme codes found. Check that the 'Funds' and 'Watchlist' tabs have a 'Scheme Code' column populated.")
    st.stop()

nav_history = fetch_nav_history(all_scheme_codes)
navs_df = pd.DataFrame(nav_history).ffill()

fund_names_map = get_names_from_codes(list(all_fund_codes | set(INDICES.values())))

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📁 Our Funds", "📈 Indices", "👀 Watchlist", "🌍 Macro Factors"])

with tab1:
    st.subheader("Our Selected Funds — Trailing Returns")
    display_fund_group(funds_df, navs_df, fund_names_map)

with tab2:
    st.subheader("Indices — Trailing Returns")
    idx_trailing = {}
    for name, code in INDICES.items():
        if code in navs_df.columns:
            idx_trailing[name] = calculate_trailing_returns(navs_df[code])
    if idx_trailing:
        idx_df = pd.DataFrame(idx_trailing).T
        final_cols = [c for c in TRAILING_COLS_ORDER if c in idx_df.columns]
        st.dataframe(style_table(idx_df[final_cols].style, '{:.2%}', 'N/A', excel_cmap), use_container_width=True)
    else:
        st.info("Index NAV data not available yet.")

with tab3:
    st.subheader("Watchlist — Trailing Returns")
    display_fund_group(watchlist_df, navs_df, fund_names_map)

with tab4:
    st.subheader("Macro Factors")
    st.caption("Nifty 50 PE and India 10Y G-Sec are pulled from the 'Macro' tab of your Google Sheet "
               "(update these manually — no reliable free public API exists for them). "
               "US 10Y Treasury, USD/INR, Gold and Silver are pulled live from Yahoo Finance.")

    macro_sheet_df = read_macro_sheet(google_sheet_id, macro_tab)
    macro_rows = []

    for factor in MACRO_FACTORS:
        if factor["source"] == "sheet":
            if factor["key"] in macro_sheet_df.columns:
                series = macro_sheet_df[factor["key"]]
            else:
                series = pd.Series(dtype=float)
                st.warning(f"Column '{factor['key']}' not found in the '{macro_tab}' tab — skipping {factor['label']}.")
        else:
            series = fetch_yf_series(factor["key"])
            if factor.get("divide"):
                series = series / factor["divide"]
        macro_rows.append(build_macro_row(factor["label"], series))

    # Nifty 50 moving averages, derived from the index price series
    nifty_price = fetch_yf_series(NIFTY_PRICE_TICKER)
    if not nifty_price.empty:
        ma50 = nifty_price.rolling(50).mean().dropna()
        ma200 = nifty_price.rolling(200).mean().dropna()
        macro_rows.append(build_macro_row("Nifty 50 — 50 Day Moving Average", ma50))
        macro_rows.append(build_macro_row("Nifty 50 — 200 Day Moving Average", ma200))
    else:
        st.warning("Could not fetch Nifty 50 price data for moving averages.")

    macro_df = pd.DataFrame(macro_rows)
    ordered_cols = ["Macro Factor", "Current",
                    "1M Ago", "1M % Chg", "3M Ago", "3M % Chg",
                    "6M Ago", "6M % Chg", "1Y Ago", "1Y % Chg"]
    macro_df = macro_df[[c for c in ordered_cols if c in macro_df.columns]]
    st.dataframe(style_macro_table(macro_df), use_container_width=True)

st.session_state.analysis_run = True