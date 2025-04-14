import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

# 读取数据
data = pd.read_csv('HSI.csv')
data['trade_date'] = pd.to_datetime(data['trade_date'])
data.sort_values('trade_date', inplace=True)

# 特征工程
data['5d_pct_chg'] = data['close'].pct_change(5).cumsum()
data['vol_mean'] = data['vol'].rolling(window=5).mean()
data['volatility'] = data['close'].rolling(window=20).std()

# 构造趋势标签
def label_trend(pct):
    if pct > 0.02:
        return 'Bull'
    elif pct < -0.02:
        return 'Bear'
    else:
        return 'Neutral'

data['trend'] = data['5d_pct_chg'].apply(label_trend)
data.dropna(inplace=True)

# 特征和目标
X = data[['open', 'close', 'vol', 'swing', 'vol_mean', 'volatility']]
y = data['trend']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 标准化
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM模型
svm = SVC(kernel='rbf')
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01]}
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# 预测与评估
y_pred = grid_search.predict(X_test_scaled)
print(classification_report(y_test, y_pred))