import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

# 1. 加载数据
df = pd.read_csv('hsi.csv')
# df['target'] = df['pct_chg'].apply(lambda x: 1 if x > 0 else 0)
df['target'] = df['pct_chg'].shift(-1).apply(lambda x: 1 if x > 0 else 0).fillna(0)
df['next_open'] = df['open'].shift(-1).fillna(df['close'].iloc[0])
# df['price_change_sign'] = (df['close'] - df['open']) / df['open']
# df['price_change_sign'] = df['price_change_sign'].apply(lambda x: 1 if x > 0 else 0)
# print(df[['price_change_sign','pct_chg','target']])
# print((df['price_change_sign'] ^ df['target']).sum())
# exit()

df['next_pct_chg'] = df['pct_chg'].shift(-1, fill_value=df['pct_chg'].iloc[-1])
# 2. 生成目标变量
# 将 pct_chg 列的正值转为 1，负值转为 0
# 计算简单移动平均线（SMA） - 5日均线和10日均线
df['SMA_5'] = df['close'].rolling(window=5).mean()  # 5日SMA
df['SMA_10'] = df['close'].rolling(window=10).mean()  # 10日SMA

# 计算指数加权移动平均线（EMA） - 5日均线和10日均线
df_reversed = df.iloc[::-1]

# Calculate EMA over the next 5 rows (in reversed order)
df_reversed['next_ema'] = df_reversed['close'].ewm(span=5, adjust=False).mean()

# Reverse it back to the original order
df_reversed['close_EMA_5'] = df_reversed['close'].ewm(span=5, adjust=False).mean()  # 5日EMA
df_reversed['pct_EMA_5'] = df_reversed['pct_chg'].ewm(span=5, adjust=False).mean()
df_reversed['open_EMA_5'] = df_reversed['open'].ewm(span=5, adjust=False).mean()
df['close_EMA_5'] = df_reversed['close_EMA_5'].iloc[::-1].values
df['pct_EMA_5'] = df_reversed['pct_EMA_5'].iloc[::-1].values
df['open_EMA_5'] = df_reversed['open_EMA_5'].iloc[::-1].values
# 3. 选择特征变量
# 可以选择有用的特征（如 open, close, high, low, swing, vol 等）
df = df.iloc[1:-1]
features = ['next_open','close','open_EMA_5','close_EMA_5']
X = df[features]
# part = df[(df['trade_date'].dt.year <= 2018 + 7) & (df['trade_date'].dt.year >= 2018)]
# X1 = part[features]
# y1 = part['pct_chg'].apply(lambda x: 1 if x > 0 else 0)
# 4. 分割数据集为训练集和测试集

# 补充: 训练集来自2018到2015,而测试集来自整个dataset

X_train, X_test, y_train, y_test = train_test_split(X, df['target'], test_size=0.4, random_state=22)

# 5. 初始化并训练决策树模型
model = DecisionTreeClassifier(random_state=21)
model.fit(X_train, y_train)

# 6. 预测测试集
y_pred = model.predict(X_test)

# # 7. 评估模型
# accuracy = accuracy_score(y_test, y_pred)

# from sklearn.model_selection import train_test_split, cross_val_score

# # 8. 如果需要，输出决策树规则
from sklearn.tree import export_text
print(export_text(model, feature_names=features))
# print(f'Accuracy: {accuracy:.4f}')
cv_scores = cross_val_score(model, X, df['target'], cv=10, scoring='accuracy')

print(f'Cross-validation scores: {cv_scores}')
print(f'Mean accuracy: {cv_scores.mean():.4f}')
print(f'Standard deviation of accuracy: {cv_scores.std():.4f}')
# from sklearn import tree
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 300)

# tree.plot_tree(model);
# fig.savefig('plottreedefault.png')