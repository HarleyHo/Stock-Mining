import time
from typing import Dict
import os
import tushare as ts
import pandas as pd

ts.set_token(os.environ['TS_TOKEN'])
pro = ts.pro_api()


def get_hsi(api=pro):
    hsi = api.index_global(ts_code='HSI', start_date='20150412', end_date='20250412')
    hsi['trade_date'] = pd.to_datetime(hsi['trade_date'], format='%Y%m%d')
    hsi.sort_values('trade_date', inplace=True)
    hsi.to_csv('HSI.csv', index=False)
    return hsi


def get_open_date(api=pro):
    dates = api.hk_tradecal(start_date='20250112', end_date='20250116', is_open='1')
    dates = dates[['cal_date']].sort_values('cal_date')
    return dates['cal_date'].tolist()


def get_daily(trade_date, api=pro):
    for _ in range(3):
        try:
            df = api.hk_daily(trade_date=trade_date)
            return df
        except Exception as e:
            print(f"API access frequency limit exceeded: {e}")
            time.sleep(1)
    return pd.DataFrame()  # Return empty DataFrame if all attempts fail


def get_hkex(api=pro):
    hkex = api.hk_basic()
    hkex.to_csv('HKEX.csv', index=False)
    return hkex


def get_ts(trade_dates, ts_codes, api=pro) -> Dict[str, pd.DataFrame]:
    # Initialize dictionary with empty DataFrames for each stock code
    ts_data: Dict[str, pd.DataFrame] = {ts_code: pd.DataFrame() for ts_code in ts_codes}

    # Iterate through each trading date
    for trade_date in trade_dates:
        df = get_daily(trade_date, api)
        if df.empty:
            print(f"No data retrieved for date {trade_date}")
            continue

        # For each stock in the daily data, append to its DataFrame
        for ts_code in ts_codes:
            # Filter rows for current ts_code
            stock_data = df[df['ts_code'] == ts_code]
            if not stock_data.empty:
                # Append to existing DataFrame
                ts_data[ts_code] = pd.concat([ts_data[ts_code], stock_data], ignore_index=True)

        print(f"Processed data for date {trade_date}")

    return ts_data


# Main execution
if __name__ == "__main__":
    # Get HSI data
    get_hsi()

    # Get HKEX basic info
    hkex = get_hkex()

    # Get trading dates
    dates = get_open_date()

    # Get stock codes
    ts_codes = hkex['ts_code'].tolist()

    # Get trading data
    stock_data = get_ts(dates, ts_codes)

    # Optional: Save each stock's DataFrame to CSV
    for ts_code, df in stock_data.items():
        if not df.empty:
            df.to_csv(f'stock_{ts_code}.csv', index=False)