import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import json
import itertools

df = pd.read_csv('processed_data.csv')


# print(df['pct_chg'])
# Convert the string representation of lists to actual lists of numbers
def a(x):
    # print(x)
    return ast.literal_eval(x)


df['pct_chg'] = df['pct_chg'].apply(a)


# ;ast.literal_eval(x)
# # Expand all lists into a single large array
# all_pct_chg_values = np.concatenate(df['pct_chg'].values)

# # Calculate the mean and standard deviation of the entire data
# mean = np.mean(all_pct_chg_values)
# std = np.std(all_pct_chg_values)

# # Standardize all lists (z-score normalization)
# def standardize_list(lst):
#     return [(x - mean) / std for x in lst]

# df['pct_chg'] = df['pct_chg'].apply(standardize_list)
# print(df['pct_chg'])
# Expand all lists into a single large array
# all_pct_chg_values = np.concatenate(df['pct_chg'].values)

# # Calculate the minimum and maximum values of the entire data
# min_val = np.min(all_pct_chg_values)
# max_val = np.max(all_pct_chg_values)

# # Normalize all lists (min-max normalization)
# def normalize_list(lst):
#     return [(x - min_val) / (max_val - min_val) for x in lst]

# df['pct_chg'] = df['pct_chg'].apply(normalize_list)
# print(df['pct_chg'])
# Prepare the data for clustering by flattening the pct_chg lists
# X = df['pct_chg'].to_list()
# Function to extract features from pct_chg
def calculate_atr(pct_chg_list, window=15):
    pct_chg_array = np.array(pct_chg_list)
    # Calculate the True Range (TR)
    tr = np.abs(np.diff(pct_chg_array))  # Difference between consecutive pct_chg values
    atr = np.convolve(tr, np.ones(window) / window, mode='valid')  # Moving average of True Range
    return np.mean(atr)


# Function to calculate ADX (Average Directional Index)
# Calculate the average value of ADX
def calculate_adx(pct_chg_list, window=15):
    pct_chg_array = np.array(pct_chg_list)

    # Calculate upward and downward directional movements
    up_move = np.maximum(0, np.diff(pct_chg_array))  # Positive movement
    down_move = np.maximum(0, -np.diff(pct_chg_array))  # Negative movement

    # Pad up_move and down_move to match the original array length (fill front with 0)
    up_move = np.pad(up_move, (1, 0), mode='constant', constant_values=0)
    down_move = np.pad(down_move, (1, 0), mode='constant', constant_values=0)

    # Calculate smoothed directional movements
    plus_di = 100 * (np.convolve(up_move, np.ones(window) / window, mode='valid') / np.convolve(np.abs(pct_chg_array),
                                                                                                np.ones(
                                                                                                    window) / window,
                                                                                                mode='valid'))
    minus_di = 100 * (
                np.convolve(down_move, np.ones(window) / window, mode='valid') / np.convolve(np.abs(pct_chg_array),
                                                                                             np.ones(window) / window,
                                                                                             mode='valid'))

    # Pad DI arrays to match the original length
    plus_di = np.pad(plus_di, (window - 1, 0), mode='constant', constant_values=0)
    minus_di = np.pad(minus_di, (window - 1, 0), mode='constant', constant_values=0)

    # Calculate ADX
    adx = np.abs(plus_di - minus_di)  # Calculate the difference between +DI and -DI
    adx = np.convolve(adx, np.ones(window) / window, mode='valid')  # Moving average of ADX
    adx = np.pad(adx, (window - 1, 0), mode='constant', constant_values=0)  # Pad ADX result to match the original array

    return np.mean(adx)  # Return the average value of ADX


def calculate_ema(pct_chg_list, span=15):
    pct_chg_array = np.array(pct_chg_list)
    # Calculate EMA using pandas ewm function
    ema = pd.Series(pct_chg_array).ewm(span=span, adjust=False).mean()
    return ema.mean()


def extract_features(pct_chg_list):
    # Convert the string list to a list of floats
    pct_chg_list = np.array(pct_chg_list)

    # Calculate statistical features
    mean = np.mean(pct_chg_list)
    std_dev = np.std(pct_chg_list)
    max_val = np.max(pct_chg_list)
    min_val = np.min(pct_chg_list)
    median = np.median(pct_chg_list)
    skewness = skew(pct_chg_list)
    kurt = kurtosis(pct_chg_list)
    adx = calculate_adx(pct_chg_list)
    atr = calculate_atr(pct_chg_list)
    ema = calculate_ema(pct_chg_list)
    # print(ema)
    return [mean, std_dev, max_val, min_val, median, skewness, kurt, adx, atr, ema]


# Apply the function to the pct_chg column
features = df['pct_chg'].apply(extract_features)

# Convert the list of features into separate columns
feature_columns = pd.DataFrame(features.tolist(),
                               columns=['mean', 'std_dev', 'max', 'min', 'median', 'skewness', 'kurtosis', 'adx', 'atr',
                                        'ema'])

# Concatenate the features back to the original dataframe
df_with_features = pd.concat([df, feature_columns], axis=1)
# print(df_with_features)
# features = ['day_of_year', 'open', 'close', 'high', 'low', 'swing', 'vol']
features = ['mean', 'std_dev', 'max', 'min', 'median', 'skewness', 'kurtosis', 'adx', 'atr', 'ema']
res = [{"rate": 0, "list": []} for i in features]
# 2. Generate all possible combinations of feature lengths
features = ['std_dev', 'min', 'kurtosis', 'atr']
X = df_with_features[features]

# 3. Data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 4. Use the elbow method to select the optimal k value
inertia = []
start = 2
limit = 20
for k in range(start, limit):  # Try cluster numbers from 1 to 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
inertia_diff = np.diff(inertia)
# print(inertia_diff)
# Calculate the second-order difference of the rate of change (acceleration of change)
inertia_diff2 = np.diff(inertia_diff)
# print(inertia_diff2)
# Find the minimum value of the second-order difference (i.e., the elbow position)
k_optimal = np.argmax(inertia_diff2) + 2 + start - 1  # Add 2 because we used second-order difference

print(f"Optimal K based on Elbow method: {k_optimal}")
# Plot the elbow method graph
plt.plot(range(start, limit), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Select the appropriate k value based on the elbow method (assuming k=3)
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df_with_features['cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_with_features['cluster'], cmap='viridis')
plt.title('KMeans Clustering (PCA-reduced data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

# 6. Output the cluster labels for each sample
# print(df[['ts_code', 'trade_date', 'cluster']].head())
from sklearn.metrics import silhouette_score

print(df_with_features)
# Calculate the silhouette score
silhouette_avg = silhouette_score(X_scaled, df_with_features['cluster'])
print(silhouette_avg)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

# Assuming your DataFrame 'df' has columns: 'first_date', 'last_date', 'atr', 'ema', 'cluster'
df = df_with_features
# Convert 'first_date' to datetime (if it's not already in datetime format)
df['first_date'] = pd.to_datetime(df['first_date'], format='%Y%m%d')

# You can optionally use PCA to reduce dimensions to 2D for better visualization
pca = PCA(n_components=2)
df[['pca1', 'pca2']] = pca.fit_transform(df[['mean', 'std_dev']])

# Create the plot
plt.figure(figsize=(10, 6))

# Scatter plot of clusters, using 'first_date' as the x-axis and PCA components as the y-axis
sns.scatterplot(data=df, x='first_date', y='pca1', hue='cluster', palette='viridis', s=100, edgecolor='k')

# Labeling the plot
plt.title('Clusters Visualized by Date Chunks')
plt.xlabel('Date')
plt.ylabel('PCA 1 (of mean and std_dev)')
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.legend(title='Cluster')

# Show plot
plt.tight_layout()
plt.show()
