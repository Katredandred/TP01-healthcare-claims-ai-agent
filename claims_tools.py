import pandas as pd
from langchain.tools import tool

# ---------------------------------------------------------
# TOOL 1: The Stacked Bar Anomaly Investigator
# ---------------------------------------------------------
@tool
def investigate_claims_spike(file_path: str) -> str:
    """
    Analyzes hospital claims data from an Excel file to find the most significant month-over-month
    dollar change in BILLED amounts, identifies drivers, and identifies the anomaly.
    """
    try:
        enrollment = pd.read_excel(file_path, sheet_name='fake enrollment')
        claims = pd.read_excel(file_path, sheet_name='fake claims')
        
        # Clean IDs and perform LEFT merge
        enrollment['Member_ID'] = enrollment['Member_ID'].astype(str).str.strip()
        claims['Member_ID'] = claims['Member_ID'].astype(str).str.strip()
        df_temp = pd.merge(claims, enrollment, on='Member_ID', how='left')
        df_temp['Region'] = df_temp['Region'].fillna('Unknown Region')
    except Exception as e:
        return f"Error loading file: {str(e)}"

    # Robust Date Parsing
    if pd.api.types.is_datetime64_any_dtype(df_temp['Service_Date']):
        df_temp['Date'] = df_temp['Service_Date']
    else:
        date_strs = df_temp['Service_Date'].astype(str).str.strip()
        df_temp['Date'] = pd.to_datetime(date_strs, format='%m/%d/%Y', errors='coerce')
        df_temp['Date'] = df_temp['Date'].fillna(pd.to_datetime(date_strs, errors='coerce'))
        
        num_vals = pd.to_numeric(df_temp['Service_Date'], errors='coerce')
        is_excel_date = num_vals.notna() & (num_vals < 100000)
        if is_excel_date.any():
            df_temp.loc[is_excel_date, 'Date'] = pd.to_datetime(
                num_vals[is_excel_date], unit='D', origin='1899-12-30'
            )
            
    df_temp = df_temp.dropna(subset=['Date'])
    df_temp['YearMonth'] = df_temp['Date'].dt.to_period('M')

    # Anomaly Detection Logic
    monthly_trend = df_temp.groupby('YearMonth')['Billed_Amt'].sum().reset_index().sort_values('YearMonth')
    if len(monthly_trend) < 2:
        return "Not enough monthly data."

    monthly_trend['Dollar_Change'] = monthly_trend['Billed_Amt'].diff()
    max_spike_idx = monthly_trend['Dollar_Change'].abs().idxmax()

    spike_month = monthly_trend.loc[max_spike_idx, 'YearMonth']
    dollar_swing = monthly_trend.loc[max_spike_idx, 'Dollar_Change']
    prev_amt = monthly_trend.loc[max_spike_idx - 1, 'Billed_Amt']
    
    spike_pct = (dollar_swing / prev_amt) * 100 if prev_amt != 0 else 0
    direction = "growth" if dollar_swing > 0 else "decline"

    # Identify drivers for the text summary
    spike_data = df_temp[df_temp['YearMonth'] == spike_month]
    drivers = spike_data.groupby(['Type', 'Region'])['Billed_Amt'].sum().reset_index().sort_values('Billed_Amt', ascending=False)
    top_driver_type = drivers.iloc[0]['Type']
    top_driver_region = drivers.iloc[0]['Region']
    top_driver_amt = drivers.iloc[0]['Billed_Amt']

    return (f"The most significant change occurred in {str(spike_month)} with a swing of ${abs(dollar_swing):,.2f} "
            f"({abs(spike_pct):.1f}% {direction}). The primary driver was '{top_driver_type}' claims in the "
            f"'{top_driver_region}' region (${top_driver_amt:,.2f}).")


# ---------------------------------------------------------
# TOOL 2: MoM Percentage Point Driver Decomposition
# ---------------------------------------------------------
@tool
def analyze_incremental_paid_claims(file_path: str) -> str:
    """
    Analyzes PAID claims month-over-month. It decomposes the drivers (Region and Type)
    as percentage points of the total month-over-month percentage change.
    """
    try:
        enrollment = pd.read_excel(file_path, sheet_name='fake enrollment')
        claims = pd.read_excel(file_path, sheet_name='fake claims')
        
        # Clean IDs and perform LEFT merge
        enrollment['Member_ID'] = enrollment['Member_ID'].astype(str).str.strip()
        claims['Member_ID'] = claims['Member_ID'].astype(str).str.strip()
        df = pd.merge(claims, enrollment, on='Member_ID', how='left')
        df['Region'] = df['Region'].fillna('Unknown Region')
    except Exception as e:
        return f"Error loading file: {str(e)}"

    # Robust Date Parsing
    if pd.api.types.is_datetime64_any_dtype(df['Service_Date']):
        df['Date'] = df['Service_Date']
    else:
        date_strs = df['Service_Date'].astype(str).str.strip()
        df['Date'] = pd.to_datetime(date_strs, format='%m/%d/%Y', errors='coerce')
        df['Date'] = df['Date'].fillna(pd.to_datetime(date_strs, errors='coerce'))
        
        num_vals = pd.to_numeric(df['Service_Date'], errors='coerce')
        is_excel_date = num_vals.notna() & (num_vals < 100000)
        if is_excel_date.any():
            df.loc[is_excel_date, 'Date'] = pd.to_datetime(
                num_vals[is_excel_date], unit='D', origin='1899-12-30'
            )
            
    df = df.dropna(subset=['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    
    unique_months = sorted(df['YearMonth'].unique())

    if len(unique_months) < 2:
        return "Not enough data months available to compare periods."

    monthly_totals = df.groupby('YearMonth')['Paid_Amt'].sum()
    pivot_df = df.pivot_table(index='YearMonth', columns=['Region', 'Type'], values='Paid_Amt', aggfunc='sum', fill_value=0)
    diff_df = pivot_df.diff()
    prev_totals = monthly_totals.shift(1)

    pct_pt_df = diff_df.div(prev_totals, axis=0) * 100
    pct_pt_df = pct_pt_df.dropna()

    plot_df = pct_pt_df.stack(['Region', 'Type']).reset_index(name='Pct_Pt_Contribution')
    
    target_month = pct_pt_df.index[-1]
    prev_month = unique_months[unique_months.index(target_month) - 1]

    total_curr = monthly_totals[target_month]
    total_prev = monthly_totals[prev_month]
    mom_pct = ((total_curr - total_prev) / total_prev) * 100 if total_prev else 0

    month_data = plot_df[plot_df['YearMonth'] == target_month].copy()
    month_data['Abs_Contribution'] = month_data['Pct_Pt_Contribution'].abs()
    top_driver_row = month_data.sort_values('Abs_Contribution', ascending=False).iloc[0]

    driver_reg = top_driver_row['Region']
    driver_typ = top_driver_row['Type']
    driver_pp = top_driver_row['Pct_Pt_Contribution']

    direction = "increase" if mom_pct > 0 else "decrease"

    return (f"In the most recent month ({str(target_month)}), total paid claims were ${total_curr:,.2f}, "
            f"which is a {abs(mom_pct):.1f}% {direction} from {str(prev_month)}. "
            f"The primary driver was '{driver_typ}' claims in the '{driver_reg}' region, "
            f"contributing {driver_pp:+.1f} percentage points to the overall {mom_pct:+.1f}% MoM change.")
