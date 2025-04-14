import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('HSI.csv')
data['trade_date'] = pd.to_datetime(data['trade_date'])
data.sort_values('trade_date', inplace=True)

# 特征工程
data['MA5'] = data['close'].rolling(window=5).mean()
data['dev_MA5'] = (data['close'] - data['MA5']) / data['MA5']
data['vol_ratio'] = data['vol'] / data['vol'].rolling(window=5).mean()
data['volatility'] = data['close'].rolling(window=20).std()
data.dropna(inplace=True)

# 特征选择
X = data[['pct_chg', 'swing', 'vol_ratio', 'volatility', 'dev_MA5']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means聚类
kmeans = KMeans(n_clusters=4, random_state=42)
data['cluster'] = kmeans.fit_predict(X_scaled)

# 评估
print(f"Silhouette Score: {silhouette_score(X_scaled, data['cluster'])}")

# 可视化
plt.scatter(data['trade_date'], data['close'], c=data['cluster'], cmap='viridis')
plt.xlabel('Date')
plt.ylabel('Close')
plt.title('HSI Trading Day Clusters')
plt.show()

# 簇特征分析
cluster_stats = data.groupby('cluster')[['pct_chg', 'swing', 'vol_ratio']].mean()
print(cluster_stats)