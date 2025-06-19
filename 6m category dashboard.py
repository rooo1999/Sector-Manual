import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="6M Mutual Fund Returns", layout="wide")
st.title("📊 6-Month Mutual Fund Returns Dashboard")

# Define category-wise scheme codes
scheme_categories = {
    "Largecap": [119551, 102885],  # Example codes: replace with actual
    "Midcap": [118834, 119433],
    "Smallcap": [119597, 125497],
    "Flexicap": [118834, 103503],
    "Multicap": [119801, 101378]
}

# Function to fetch NAV data from MFAPI
@st.cache_data
def fetch_nav(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json().get("data", [])
        df = pd.DataFrame(data)
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        df = df.dropna()
        df = df.sort_values('date').reset_index(drop=True)
        return df
    else:
        return pd.DataFrame()

# Function to calculate 6-month return
def calculate_6m_return(df):
    if df.empty or df.shape[0] < 130:
        return None
    today = df['date'].max()
    past_date = today - pd.DateOffset(months=6)
    recent_nav = df[df['date'] == df['date'].max()]['nav'].values[0]
    past_navs = df[df['date'] <= past_date]
    if not past_navs.empty:
        past_nav = past_navs.iloc[-1]['nav']
        return round(((recent_nav - past_nav) / past_nav) * 100, 2)
    return None

# Sidebar for selecting category
selected_category = st.selectbox("Select Category", list(scheme_categories.keys()))

# Show codes in that category
st.write(f"### Funds in {selected_category} Category")
scheme_codes = scheme_categories[selected_category]

results = []

for code in scheme_codes:
    nav_df = fetch_nav(code)
    six_month_return = calculate_6m_return(nav_df)
    if six_month_return is not None:
        name_url = f"https://api.mfapi.in/mf/{code}"
        fund_name = requests.get(name_url).json().get("meta", {}).get("scheme_name", f"Scheme {code}")
        results.append({"Scheme Name": fund_name, "Scheme Code": code, "6M Return (%)": six_month_return})

# Convert to DataFrame and display
if results:
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="6M Return (%)", ascending=False)
    st.dataframe(result_df, use_container_width=True)
else:
    st.warning("No data available for the selected category.")
