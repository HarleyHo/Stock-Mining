import time
from typing import Dict
import os
import tushare as ts
import pandas as pd
from utils import setup_logger

logger = setup_logger()

# Set Tushare token and initialize pro API
ts.set_token(os.environ['TS_TOKEN'])
pro = ts.pro_api()

# Directory to store stock data
OUTPUT_DIR = "stock_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def get_hsi(api=pro):
    """
    Retrieve HSI index data if not already present.
    Returns: HSI DataFrame
    """
    hsi_file = 'HSI.csv'
    if os.path.exists(hsi_file):
        return pd.read_csv(hsi_file)

    hsi = api.index_global(ts_code='HSI', start_date='20150412', end_date='20250412')
    hsi['trade_date'] = pd.to_datetime(hsi['trade_date'], format='%Y%m%d')
    hsi.sort_values('trade_date', inplace=True)
    hsi.to_csv(hsi_file, index=False)


def get_open_date(api=pro):
    """
    Fetch open trading dates.
    Returns: List of trading dates
    """
    dates = api.hk_tradecal(start_date='20241112', end_date='20250412', is_open='1')
    return dates[['cal_date']].sort_values('cal_date')['cal_date'].tolist()


def get_daily(trade_date, api=pro):
    """
    Retrieve daily trading data with retry logic.
    Args:
        trade_date: Date to fetch data for
        api: Tushare API instance
    Returns: Daily data DataFrame
    """
    for _ in range(3):
        try:
            stock = api.hk_daily(trade_date=trade_date)
            logger.info(f"Daily data of {trade_date} obtain successfully.")
            return stock
        except Exception:
            time.sleep(1)
    return pd.DataFrame()


def get_hkex(api=pro):
    """
    Retrieve HKEX basic info if not already present.
    Returns: HKEX DataFrame
    """
    hkex_file = 'HKEX.csv'
    if os.path.exists(hkex_file):
        return pd.read_csv(hkex_file)

    hkex = api.hk_basic()
    hkex.to_csv(hkex_file, index=False)
    return hkex


def get_ts(trade_dates, ts_codes, api=pro) -> Dict[str, pd.DataFrame]:
    """
    Fetch trading data for specified stocks and dates.
    Args:
        trade_dates: List of trading dates
        ts_codes: List of stock codes
        api: Tushare API instance
    Returns: Dictionary mapping stock codes to their DataFrames
    """
    ts_data: Dict[str, pd.DataFrame] = {ts_code: pd.DataFrame() for ts_code in ts_codes}

    for trade_date in trade_dates:
        df = get_daily(trade_date, api)
        if df.empty:
            continue

        for ts_code in ts_codes:
            stock_data = df[df['ts_code'] == ts_code]
            if not stock_data.empty:
                ts_data[ts_code] = pd.concat([ts_data[ts_code], stock_data], ignore_index=True)

    return ts_data


# Execute data retrieval
get_hsi()
hkex = get_hkex()
dates = get_open_date()
ts_codes = hkex['ts_code'].tolist()
stock_data = get_ts(dates, ts_codes)

# Save stock data to individual CSV files in output directory
for ts_code, df in stock_data.items():
    if not df.empty:
        df.to_csv(os.path.join(OUTPUT_DIR, f'{ts_code}.csv'), index=False)
