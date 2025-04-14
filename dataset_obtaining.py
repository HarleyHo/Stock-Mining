import time

import tushare as ts
import pandas as pd

ts.set_token('da90991b4f0659b48d202e536f7e400b946027197b8276e82ae48971')
pro = ts.pro_api()

def get_hsi(api = pro):
    hsi = api.index_global(ts_code='HSI', start_date='20150412', end_date='20250412')
    hsi['trade_date'] = pd.to_datetime(hsi['trade_date'], format='%Y%m%d')
    hsi.sort_values('trade_date', inplace=True)
    hsi.to_csv('HSI.csv', index=False)

def get_open_date(api = pro):
    dates = api.hk_tradecal(start_date='20250112', end_date='20250116', is_open='1')
    dates = dates[['cal_date']]
    dates = dates.sort_values('cal_date')
    return dates['cal_date'].tolist()

def get_daily(trade_date, api = pro):
    for _ in range(3):
        try:
            df = api.hk_daily(trade_date=trade_date)
            print(df.head())
        except Exception as e:
            print("API access frequency limit exceeded.")
            time.sleep(1)
        else:
            return df

def get_hkex(api = pro):
    hkex = api.hk_basic()
    hkex.to_csv('HKEX.csv', index=False)

def get_ts(trade_dates, api = pro):
    for trade_date in trade_dates:
        df = get_daily(trade_date)


dates = get_open_date()
get_ts(dates)