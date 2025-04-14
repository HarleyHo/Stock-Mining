import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas_ta as ta

# 读取数据
df = pd.read_csv('HSI.csv', parse_dates=["trade_date"])

# 特征工程
df.ta.ema(10, append=True)
df.ta.ema(30, append=True)

# df.ta.atr()
# df.ta.adx()
# df.ta.rsi(14)
print(df.head())

# # 特征和目标
# X = df[['open', 'close', 'high', 'low', 'pct_chg', 'vol', 'swing', 'MA5', 'RSI']]
# y = df['y']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#
# # 决策树模型
# clf = DecisionTreeClassifier()
# param_grid = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
# grid_search = GridSearchCV(clf, param_grid, cv=5)
# grid_search.fit(X_train, y_train)
#
# # 预测与评估
# y_pred = grid_search.predict(X_test)
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# print(confusion_matrix(y_test, y_pred))
