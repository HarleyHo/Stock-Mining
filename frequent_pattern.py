import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 读取数据
data = pd.read_csv('HSI.csv')
data['trade_date'] = pd.to_datetime(data['trade_date'])
data.sort_values('trade_date', inplace=True)

# 离散化
def discretize_pct_chg(x):
    if x > 2:
        return 'Big_Up'
    elif x > 0:
        return 'Small_Up'
    elif x > -2:
        return 'Small_Down'
    else:
        return 'Big_Down'

def discretize_swing(x):
    if x > 3:
        return 'High_Swing'
    elif x > 1:
        return 'Mid_Swing'
    else:
        return 'Low_Swing'

vol_quantiles = data['vol'].quantile([0.25, 0.75])
def discretize_vol(x):
    if x > vol_quantiles[0.75]:
        return 'High_Vol'
    elif x > vol_quantiles[0.25]:
        return 'Mid_Vol'
    else:
        return 'Low_Vol'

data['pct_chg_cat'] = data['pct_chg'].apply(discretize_pct_chg)
data['swing_cat'] = data['swing'].apply(discretize_swing)
data['vol_cat'] = data['vol'].apply(discretize_vol)

# 构造事务集
transactions = data[['pct_chg_cat', 'swing_cat', 'vol_cat']].values.tolist()

# 独热编码
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori算法
frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
rules = rules[rules['lift'] > 1]

# 输出规则
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])