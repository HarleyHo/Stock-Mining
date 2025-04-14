import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('HSI.csv', parse_dates=["trade_date"])

# 静态折线图（使用matplotlib）
plt.figure(figsize=(12, 6))
plt.plot(df['trade_date'], df['close'], label='Close Price', color='blue')
plt.title('Hang Seng Index (HSI) Close Price (2015-2025)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

