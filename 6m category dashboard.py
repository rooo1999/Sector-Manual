import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="6M Returns Since 2015", layout="wide")
st.title("📈 Mutual Fund 6-Month Returns (Rolling from Custom Start Date)")

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

# Calculate absolute 6M return over each 6M period from start_date
def calculate_6m_returns(df, start_date):
    results = []
    current_start = start_date
    while True:
        current_end = current_start + relativedelta(months=6)

        # Filter NAVs
        start_nav = df[df['date'] <= current_start].tail(1)
        end_nav = df[df['date'] <= current_end].tail(1)

        if not start_nav.empty and not end_nav.empty:
            nav_start = start_nav['nav'].values[0]
            nav_end = end_nav['nav'].values[0]
            abs_return = round(((nav_end - nav_start) / nav_start) * 100, 2)
            results.append({
                "Period": f"{current_start.strftime('%d-%b-%Y')} to {current_end.strftime('%d-%b-%Y')}",
                "6M Return (%)": abs_return
            })
            current_start = current_end
        else:
            break
    return results

# Display results
st.write(f"### 6-Month Returns for {selected_category} Funds")
scheme_codes = scheme_categories[selected_category]

for code in scheme_codes:
    nav_df = fetch_nav(code)
    if nav_df.empty:
        continue

    fund_name = requests.get(f"https://api.mfapi.in/mf/{code}").json().get("meta", {}).get("scheme_name", "Unnamed Fund")

    returns = calculate_6m_returns(nav_df, pd.to_datetime(start_date))
    returns_df = pd.DataFrame(returns)
    if not returns_df.empty:
        st.subheader(f"📌 {fund_name}")
        st.dataframe(returns_df, use_container_width=True)
    else:
        st.warning(f"No return data available for {fund_name}")
