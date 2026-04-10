import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta

# ==========================================
# 1. YOUR SIF MAPPING
# ==========================================
sif_mapping = {
    "ICICI": "INF109K30034",
    # Add your actual ISINs here
    "Test Person 2": "INF846K01NG6", 
}

# ==========================================
# 2. SMART SIF SCRAPER (No html5lib needed)
# ==========================================
def fetch_sif_data(isin):
    url = "https://www.amfiindia.com/sif/latest-nav/nav-history"
    session = requests.Session()
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        # Step 1: GET request to load the form
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Calculate Dates
        end_date = datetime.today()
        start_date = end_date - timedelta(days=400)
        start_str = start_date.strftime('%d-%b-%Y')
        end_str = end_date.strftime('%d-%b-%Y')
        
        # Step 2: Dynamically build the POST payload
        payload = {}
        for inp in soup.find_all('input'):
            name = inp.get('name')
            if not name: 
                continue
            
            value = inp.get('value', '')
            name_lower = name.lower()
            
            if 'isin' in name_lower:
                payload[name] = isin
            elif 'fromdate' in name_lower or ('from' in name_lower and 'date' in name_lower):
                payload[name] = start_str
            elif 'todate' in name_lower or ('to' in name_lower and 'date' in name_lower):
                payload[name] = end_str
            else:
                payload[name] = value
                
        # Step 3: POST the form submission
        post_response = session.post(url, data=payload, headers=headers, timeout=15)
        post_response.raise_for_status()
        
        # Step 4: MANUALLY EXTRACT TABLE USING BEAUTIFULSOUP (Bypassing pd.read_html)
        post_soup = BeautifulSoup(post_response.text, 'html.parser')
        
        # Find all tables on the resulting page
        tables = post_soup.find_all('table')
        target_table = None
        
        # Locate the table that actually contains the NAV data
        for table in tables:
            text = table.get_text().lower()
            if 'date' in text and 'nav' in text:
                target_table = table
                break
                
        if not target_table:
            return None, "No data table found on the AMFI response page."

        # Parse the table headers
        raw_rows = target_table.find_all('tr')
        if not raw_rows:
            return None, "Table found, but it has no rows."

        # Extract headers from the first row (sometimes they use <th>, sometimes <td>)
        headers = [h.get_text(strip=True).lower() for h in raw_rows[0].find_all(['th', 'td'])]
        
        # Parse the table data
        data = []
        for row in raw_rows[1:]: # Skip header row
            cols = row.find_all('td')
            if len(cols) == len(headers):
                data.append([c.get_text(strip=True) for c in cols])
                
        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)
        
        # Identify the correct columns
        date_col = next((col for col in df.columns if 'date' in col), None)
        nav_col = next((col for col in df.columns if 'nav' in col), None)
        
        if not date_col or not nav_col:
            return None, "Could not identify Date or NAV columns."
            
        # Clean and convert the data
        df = df.rename(columns={date_col: 'Date', nav_col: 'NAV'})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['NAV'] = pd.to_numeric(df['NAV'], errors='coerce')
        
        df = df.dropna(subset=['Date', 'NAV']).sort_values('Date').reset_index(drop=True)
        
        if df.empty:
            return None, "Table parsed, but no valid numeric data was found."
            
        return df, "Success"
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# ==========================================
# 3. PERFORMANCE CALCULATOR
# ==========================================
def calculate_metrics(df):
    latest_date = df['Date'].iloc[-1]
    latest_nav = df['NAV'].iloc[-1]
    
    def get_nav_days_ago(target_days):
        target_date = latest_date - timedelta(days=target_days)
        closest_idx = (df['Date'] - target_date).abs().idxmin()
        return df.loc[closest_idx, 'NAV']

    try:
        nav_30d = get_nav_days_ago(30)
        nav_90d = get_nav_days_ago(90)
        nav_365d = get_nav_days_ago(365)
        
        mom = ((latest_nav / nav_30d) - 1) * 100
        qoq = ((latest_nav / nav_90d) - 1) * 100
        yoy = ((latest_nav / nav_365d) - 1) * 100
        
        return latest_nav, latest_date, mom, qoq, yoy
    except Exception:
        return latest_nav, latest_date, None, None, None

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("-" * 80)
    print(f"{'MANAGER/SIF':<20} | {'ISIN':<15} | {'LATEST NAV':<12} | {'MoM %':<8} | {'QoQ %':<8} | {'YoY %':<8}")
    print("-" * 80)

    for name, isin in sif_mapping.items():
        df, status = fetch_sif_data(isin)
        
        if df is not None:
            latest_nav, latest_date, mom, qoq, yoy = calculate_metrics(df)
            
            f_nav = f"₹{latest_nav:.4f}"
            f_mom = f"{mom:.2f}%" if mom is not None else "N/A"
            f_qoq = f"{qoq:.2f}%" if qoq is not None else "N/A"
            f_yoy = f"{yoy:.2f}%" if yoy is not None else "N/A"
            
            print(f"{name:<20} | {isin:<15} | {f_nav:<12} | {f_mom:<8} | {f_qoq:<8} | {f_yoy:<8}")
        else:
            print(f"{name:<20} | {isin:<15} | {'FAILED':<12} | {'-':<8} | {'-':<8} | {'-':<8}  -> {status}")

    print("-" * 80)