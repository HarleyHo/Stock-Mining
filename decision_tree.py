import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

# 1. Load data
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
# 2. Generate target variable
# Convert positive values in pct_chg column to 1, negative values to 0
# Calculate Simple Moving Average (SMA) - 5-day and 10-day
df['SMA_5'] = df['close'].rolling(window=5).mean()  # 5-day SMA
df['SMA_10'] = df['close'].rolling(window=10).mean()  # 10-day SMA

# Calculate Exponential Moving Average (EMA) - 5-day and 10-day
df_reversed = df.iloc[::-1]

# Calculate EMA over the next 5 rows (in reversed order)
df_reversed['next_ema'] = df_reversed['close'].ewm(span=5, adjust=False).mean()

# Reverse it back to the original order
df_reversed['close_EMA_5'] = df_reversed['close'].ewm(span=5, adjust=False).mean()  # 5-day EMA
df_reversed['pct_EMA_5'] = df_reversed['pct_chg'].ewm(span=5, adjust=False).mean()
df_reversed['open_EMA_5'] = df_reversed['open'].ewm(span=5, adjust=False).mean()
df['close_EMA_5'] = df_reversed['close_EMA_5'].iloc[::-1].values
df['pct_EMA_5'] = df_reversed['pct_EMA_5'].iloc[::-1].values
df['open_EMA_5'] = df_reversed['open_EMA_5'].iloc[::-1].values
# 3. Select feature variables
# Select useful features (e.g., open, close, high, low, swing, vol, etc.)
df = df.iloc[1:-1]
features = ['next_open','close','open_EMA_5','close_EMA_5']
X = df[features]
# part = df[(df['trade_date'].dt.year <= 2018 + 7) & (df['trade_date'].dt.year >= 2018)]
# X1 = part[features]
# y1 = part['pct_chg'].apply(lambda x: 1 if x > 0 else 0)
# 4. Split dataset into training and testing sets

# Note: Training set comes from 2018 to 2015, while testing set comes from the entire dataset

X_train, X_test, y_train, y_test = train_test_split(X, df['target'], test_size=0.4, random_state=22)

# 5. Initialize and train the decision tree model
model = DecisionTreeClassifier(random_state=21)
model.fit(X_train, y_train)

# 6. Predict on the test set
y_pred = model.predict(X_test)

# # 7. Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)

# from sklearn.model_selection import train_test_split, cross_val_score

# 8. If needed, output decision tree rules
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