import pandas as pd
import tushare as ts
import os
import matplotlib.pyplot as plt

# 读取数据
ts.set_token(os.environ['TS_TOKEN'])
pro = ts.pro_api()
df = pro.hk_daily(ts_code="00417.HK", start_date="20250113", end_date="20250201")
print(df.head())