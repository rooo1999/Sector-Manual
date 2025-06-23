import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Mutual Fund MoM Returns", layout="wide")
st.title("📊 Mutual Fund Month-on-Month (MoM) Returns Dashboard")

# ---- Function to get NAV data from MFAPI ----
@st.cache_data(ttl=86400)
def get_nav_data(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        scheme_name = data['meta']['scheme_name']
        navs = data['data']
        df = pd.DataFrame(navs)
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df = df.dropna().sort_values('date').reset_index(drop=True)
        return scheme_name, df
    else:
        return None, None

# ---- Function to get month-end NAVs ----
def get_month_end_navs(df):
    df['month'] = df['date'].dt.to_period('M')
    month_ends = df.groupby('month').apply(lambda x: x.loc[x['date'].idxmax()])
    month_ends = month_ends.reset_index(drop=True)
    return month_ends[['date', 'nav']]

# ---- Function to calculate MoM returns ----
def calculate_mom_returns(nav_df):
    nav_df['return'] = nav_df['nav'].pct_change() * 100
    nav_df['month'] = nav_df['date'].dt.strftime('%b-%Y')
    return nav_df[['month', 'return']].iloc[1:]  # Skip first NaN return

# ---- User input ----
default_codes = "105804,
125350,
147944,
145677,
152130,
146127,
105989,
146193,
103360,
130502,
108097,
106823,
145139,
147920,
152612,
152003,
150912,
153198,
152232,
113177,
149020,
100177,
152108,
125494,
100795,
145208,
152940,
129647,
148617, "
scheme_codes_input = st.text_input("Enter scheme codes (comma-separated):", default_codes)
scheme_codes = [code.strip() for code in scheme_codes_input.split(',') if code.strip().isdigit()]

# ---- Processing ----
all_returns = pd.DataFrame()

for code in scheme_codes:
    scheme_name, df = get_nav_data(code)
    if df is not None:
        month_end_navs = get_month_end_navs(df)
        returns_df = calculate_mom_returns(month_end_navs)
        returns_df.rename(columns={'return': scheme_name}, inplace=True)
        if all_returns.empty:
            all_returns = returns_df
        else:
            all_returns = pd.merge(all_returns, returns_df, on='month', how='outer')

# ---- Display Results ----
if not all_returns.empty:
    all_returns = all_returns.sort_values(by='month')
    all_returns = all_returns.set_index('month').round(2)
    st.dataframe(all_returns.style.background_gradient(cmap="RdYlGn", axis=1))
else:
    st.warning("No valid data found. Please check the scheme codes entered.")
