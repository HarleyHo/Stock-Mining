import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Loading
file_path = 'hsi.csv'  # Assuming the data file path
data = pd.read_csv(file_path)

# 2. Data Processing: Extract required columns
data['pct_chg'] = data['pct_chg'].fillna(0)  # Fill missing values
data['swing'] = data['swing'].fillna(0)
data['vol'] = data['vol'].fillna(0)


# 3. Feature Binning: Equal-frequency binning using quantiles
def quantile_binning(df, column, n_bins=3):
    """Bin the data of a column into categories using quantiles"""
    labels = ['Low', 'Medium', 'High']  # Define bin labels
    bins = pd.qcut(df[column], q=n_bins, labels=labels)  # Equal-frequency binning
    return bins


data['pct_chg_bin'] = quantile_binning(data, 'pct_chg')  # Binning for pct_chg (percentage change)
data['swing_bin'] = quantile_binning(data, 'swing')  # Binning for swing (amplitude)
data['vol_bin'] = quantile_binning(data, 'vol')  # Binning for vol (volume)


# 4. Convert data into a format suitable for the Apriori algorithm
df_apriori = pd.DataFrame()
df_apriori['pct_chg'] = data['pct_chg_bin']
df_apriori['swing'] = data['swing_bin']
df_apriori['vol'] = data['vol_bin']

# Encoding categorical data into binary (0/1)
df_apriori_encoded = pd.get_dummies(df_apriori)

# 5. Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df_apriori_encoded, min_support=0.05, use_colnames=True)

# 6. Extract association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# 7. Filter rules based on support and confidence thresholds
filtered_rules = rules[(rules['support'] >= 0.05) & (rules['confidence'] >= 0.5)]

# 8. Output filtered rules
print("Association rules that meet the threshold:")
print(filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# 9. Optional: Save the filtered rules to a file
filtered_rules.to_csv('filtered_association_rules.csv', index=False)

# 10. Visualization: Visualize frequent itemsets (support vs. length of itemset)
plt.figure(figsize=(10, 6))
support = frequent_itemsets['support']
itemset_length = frequent_itemsets['itemsets'].apply(lambda x: len(x))
plt.scatter(itemset_length, support, alpha=0.6)
plt.title('Frequent Itemsets: Support vs Itemset Length')
plt.xlabel('Itemset Length')
plt.ylabel('Support')
plt.show()

# 11. Visualization: Visualize the association rules based on lift
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_rules, x='support', y='lift', hue='confidence', palette='viridis', size='confidence', sizes=(20, 200), legend=False)
plt.title('Association Rules: Support vs Lift')
plt.xlabel('Support')
plt.ylabel('Lift')
plt.show()