import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="6M Returns Since 2015", layout="wide")
st.title("📈 Mutual Fund 6-Month Returns (End-of-Month Based Periods)")

# Predefined mutual fund scheme codes by category
scheme_categories = {
    "Largecap": [119551, 102885],
    "Midcap": [118834, 119433],
    "Smallcap": [119597, 125497],
    "Flexicap": [118834, 103503],
    "Multicap": [119801, 101378]
}

# User input
selected_category = st.selectbox("Select Category", list(scheme_categories.keys()))
start_date = st.date_input("Select Starting Date", value=datetime(2015, 3, 31))

# Function to fetch NAV data
@st.cache_data
def fetch_nav(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json().get("data", [])
        df = pd.DataFrame(data)
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        df = df.dropna().sort_values('date').reset_index(drop=True)
        return df
    return pd.DataFrame()

# Function to get last day of month
def end_of_month(date):
    return (date + relativedelta(months=1, day=1)) - pd.Timedelta(days=1)

# Build all 6M periods using "last day of 6th month"
def get_periods(start_date, end_date):
    periods = []
    start = pd.Timestamp(start_date)
    while True:
        six_months_later = start + relativedelta(months=6)
        end = end_of_month(six_months_later)
        if end > end_date:
            break
        periods.append((start, end))
        start = end  # move to end of last period
    return periods

# Function to get return
def get_return(df, start, end):
    if df.empty:
        return None
    start_nav = df[df['date'] <= start].tail(1)
    end_nav = df[df['date'] <= end].tail(1)
    if not start_nav.empty and not end_nav.empty:
        nav_start = start_nav['nav'].values[0]
        nav_end = end_nav['nav'].values[0]
        return round(((nav_end - nav_start) / nav_start) * 100, 2)
    return None

# Main logic
scheme_codes = scheme_categories[selected_category]
all_returns = {}
period_labels = []

# Get the global common max date across all schemes
latest_end_dates = []
for code in scheme_codes:
    nav_df = fetch_nav(code)
    if not nav_df.empty:
        latest_end_dates.append(nav_df['date'].max())

global_end = min(latest_end_dates) if latest_end_dates else datetime.today()
periods = get_periods(start_date, global_end)
period_labels = [f"{s.strftime('%d-%b-%Y')} to {e.strftime('%d-%b-%Y')}" for s, e in periods]

# For each scheme
for code in scheme_codes:
    nav_df = fetch_nav(code)
    if nav_df.empty:
        continue

    try:
        fund_name = requests.get(f"https://api.mfapi.in/mf/{code}").json().get("meta", {}).get("scheme_name", f"Scheme {code}")
    except:
        fund_name = f"Scheme {code}"

    fund_returns = []
    for start, end in periods:
        r = get_return(nav_df, start, end)
        fund_returns.append(r)

    all_returns[fund_name] = fund_returns

# Combine into one table
returns_df = pd.DataFrame(all_returns, index=period_labels).T
returns_df.index.name = "Fund"
returns_df.columns.name = "6M Periods"

# Show
if not returns_df.empty:
    st.write(f"### Combined 6-Month Returns Table for {selected_category}")
    st.dataframe(returns_df, use_container_width=True)
else:
    st.warning("No data available for the selected category.")
