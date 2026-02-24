import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="ISIN Performance Tracker", layout="wide")

# ==========================================
# 1. MAP NAMES TO ISINs HERE
# ==========================================
# Replace the placeholder 'INF...' strings with the actual ISINs you possess.
sif_mapping = {
    "SBI Magnum Hybrid L/S": "INF200K30015",      # <-- Replace with actual ISIN
}

# ==========================================
# 2. STEP 1: TRANSLATE ISIN TO SCHEME CODE
# ==========================================
@st.cache_data(ttl=86400) # Cache this for 24 hours so it runs fast
def get_isin_to_scheme_mapping():
    """Fetches the daily AMFI text file and maps every ISIN to its Scheme Code."""
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    try:
        response = requests.get(url, timeout=10)
        lines = response.text.split('\n')
        
        mapping = {}
        for line in lines:
            parts = line.split(';')
            # Check if row is actual data (starts with numeric scheme code)
            if len(parts) >= 6 and parts[0].isdigit():
                scheme_code = parts[0].strip()
                isin_growth = parts[1].strip()
                isin_reinv = parts[2].strip()
                
                # Map both possible ISIN columns to the scheme code
                if isin_growth and isin_growth != '-':
                    mapping[isin_growth] = scheme_code
                if isin_reinv and isin_reinv != '-':
                    mapping[isin_reinv] = scheme_code
                    
        return mapping
    except Exception as e:
        st.error(f"Failed to fetch AMFI Master List: {e}")
        return {}

# ==========================================
# 3. STEP 2: FETCH HISTORICAL DATA
# ==========================================
@st.cache_data(ttl=3600) 
def fetch_historical_nav(scheme_code):
    """Fetches trailing 1.5 years of NAV data for YoY calculation."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=400) 
    
    start_str = start_date.strftime('%d-%b-%Y')
    end_str = end_date.strftime('%d-%b-%Y')
    
    url = f"http://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx?tp=1&frmdt={start_str}&todt={end_str}"
    
    try:
        df = pd.read_csv(url, sep=';', on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        df = df[df['Scheme Code'] == int(scheme_code)]
        
        if df.empty: return None, None
            
        scheme_name = df['Scheme Name'].iloc[0]
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
        df['Net Asset Value'] = pd.to_numeric(df['Net Asset Value'], errors='coerce')
        
        df = df.sort_values('Date').dropna(subset=['Net Asset Value']).reset_index(drop=True)
        return df, scheme_name
        
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return None, None

# ==========================================
# 4. PERFORMANCE CALCULATIONS
# ==========================================
def calculate_metrics(df):
    latest_date = df['Date'].iloc[-1]
    latest_nav = df['Net Asset Value'].iloc[-1]
    
    def get_nav_days_ago(days):
        target_date = latest_date - timedelta(days=days)
        closest = df.iloc[(df['Date'] - target_date).abs().argsort()[:1]]
        if not closest.empty:
            return closest['Net Asset Value'].values[0]
        return None

    nav_30d = get_nav_days_ago(30)
    nav_90d = get_nav_days_ago(90)
    nav_365d = get_nav_days_ago(365)
    
    mom = ((latest_nav / nav_30d) - 1) * 100 if nav_30d else None
    qoq = ((latest_nav / nav_90d) - 1) * 100 if nav_90d else None
    yoy = ((latest_nav / nav_365d) - 1) * 100 if nav_365d else None
    
    return latest_nav, latest_date, mom, qoq, yoy

# ==========================================
# 5. STREAMLIT DASHBOARD UI
# ==========================================
st.title("📊 ISIN-Based SIF Performance Tracker")
st.markdown("Translates ISINs to AMFI Scheme Codes to track trailing MoM, QoQ, and YoY NAV returns.")

# Load the master dictionary in the background
isin_to_scheme_map = get_isin_to_scheme_mapping()

selected_name = st.selectbox("Select Manager/SIF Profile:", list(sif_mapping.keys()))
target_isin = sif_mapping[selected_name]

st.caption(f"Target ISIN: `{target_isin}`")

if st.button("Fetch Performance"):
    if target_isin not in isin_to_scheme_map:
        st.error(f"ISIN {target_isin} was not found in the active AMFI database.")
    else:
        scheme_code = isin_to_scheme_map[target_isin]
        
        with st.spinner(f"ISIN matched to Scheme Code {scheme_code}. Fetching historical data..."):
            df, scheme_name = fetch_historical_nav(scheme_code)
            
            if df is not None and not df.empty:
                latest_nav, latest_date, mom, qoq, yoy = calculate_metrics(df)
                
                st.success(f"**Fund Name:** {scheme_name}")
                
                # Metrics Row
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Latest NAV", f"₹{latest_nav:.4f}", f"As of {latest_date.strftime('%d %b %Y')}")
                
                def fmt(val): return f"{val:.2f}%" if val is not None else "N/A"
                col2.metric("MoM (30d)", fmt(mom), fmt(mom))
                col3.metric("QoQ (90d)", fmt(qoq), fmt(qoq))
                col4.metric("YoY (365d)", fmt(yoy), fmt(yoy))
                
                # Chart
                st.markdown("---")
                st.subheader("1-Year NAV Trend")
                fig = px.line(df, x='Date', y='Net Asset Value', template="plotly_white")
                fig.update_traces(line_color='#1E88E5', line_width=2)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("Historical data could not be fetched for this fund.")