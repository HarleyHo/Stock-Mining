import os
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Configuration constants
DATA_DIR = "stock_data"
HKEX_FILE = "HKEX.csv"
OUTPUT_DIR = "output"
MAX_CLUSTERS = 10
MIN_CLUSTER_SIZE = 3
FIGURE_DPI = 300

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_stock_data(data_dir: str = DATA_DIR, hkex_file: str = HKEX_FILE) -> Dict[str, pd.DataFrame]:
    """
    Load stock data from CSV files based on ts_codes in HKEX.csv.

    Args:
        data_dir: Directory containing stock CSV files.
        hkex_file: Path to HKEX.csv containing ts_codes.

    Returns:
        Dictionary mapping ts_code to its DataFrame.

    Raises:
        FileNotFoundError: If HKEX.csv is not found.
    """
    print("Loading stock data...")
    if not os.path.exists(hkex_file):
        raise FileNotFoundError(f"HKEX file not found: {hkex_file}")

    hkex = pd.read_csv(hkex_file)
    stock_data = {}
    for ts_code in hkex['ts_code']:
        file_path = os.path.join(data_dir, f"{ts_code}.csv")
        if os.path.exists(file_path):
            try:
                stock_data[ts_code] = pd.read_csv(file_path)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
        else:
            pass

    print(f"Loaded {len(stock_data)} stock datasets.")
    return stock_data


def calculate_features(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Calculate clustering features for a single stock.

    Args:
        df: DataFrame with stock data, including 'pct_chg' column.

    Returns:
        Series with 'avg_pct_chg' and 'volatility', or None if invalid.
    """
    try:
        if df.empty or 'pct_chg' not in df.columns:
            return None

        features = {
            'avg_pct_chg': df['pct_chg'].mean(),
            'volatility': df['pct_chg'].std()
        }
        if any(pd.isna(val) or np.isinf(val) for val in features.values()):
            return None

        return pd.Series(features)
    except Exception as e:
        print(f"Feature calculation failed: {e}")
        return None


def prepare_clustering_data(stock_data: Dict[str, pd.DataFrame]):
    """
    Prepare data for clustering by computing features.

    Args:
        stock_data: Dictionary mapping ts_code to stock DataFrame.

    Returns:
        Tuple of feature DataFrame and list of ts_codes.

    Raises:
        ValueError: If no valid features are available.
    """
    print("Preparing clustering data...")
    features_list = []
    ts_codes = []

    for ts_code, df in stock_data.items():
        feature_row = calculate_features(df)
        if feature_row is not None:
            features_list.append(feature_row)
            ts_codes.append(ts_code)

    if not features_list:
        raise ValueError("No valid features available for clustering.")

    feature_df = pd.DataFrame(features_list, index=ts_codes)
    feature_df = feature_df.dropna()
    feature_df = feature_df[~feature_df.apply(lambda x: np.any(np.isinf(x)), axis=1)]

    print(f"Prepared {len(feature_df)} stocks for clustering.")
    return feature_df, ts_codes


def plot_dendrogram(X: np.ndarray, max_clusters: int = MAX_CLUSTERS) -> int:
    """
    Plot a dendrogram to estimate the optimal number of clusters.

    Args:
        X: Scaled feature array.
        max_clusters: Maximum number of clusters to consider.

    Returns:
        Estimated optimal number of clusters.
    """
    print("Plotting dendrogram...")
    linked = linkage(X, method='ward')

    plt.figure(figsize=(10, 7))
    dendrogram(linked, truncate_mode='lastp', p=max_clusters, show_contracted=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Cluster Size')
    plt.ylabel('Distance')

    distances = linked[:, 2]
    gaps = np.diff(distances[::-1])
    optimal_k = len(gaps) - np.argmax(gaps) + 1

    plt.axhline(y=distances[-optimal_k], color='r', linestyle='--',
                label=f'Optimal k={optimal_k}')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dendrogram.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()

    return optimal_k


def plot_clusters(result: pd.DataFrame, selected_clusters: list) -> None:
    """
    Visualize clustering results as a scatter plot, highlighting selected stocks.

    Args:
        result: DataFrame with ts_code, cluster, avg_pct_chg, and volatility.
        selected_clusters: List of selected cluster indices.
    """
    print("Plotting clusters...")
    # Create a column to mark selected stocks
    result['is_selected'] = result['cluster'].isin(selected_clusters)

    plt.figure(figsize=(12, 8))

    # Plot non-selected stocks with lower opacity
    sns.scatterplot(
        data=result[~result['is_selected']],
        x='avg_pct_chg',
        y='volatility',
        hue='cluster',
        style='cluster',
        palette='deep',
        size='cluster',
        sizes=(50, 150),
        alpha=0.5,
        legend='full'
    )

    # Plot selected stocks with distinct style
    sns.scatterplot(
        data=result[result['is_selected']],
        x='avg_pct_chg',
        y='volatility',
        hue='cluster',
        style='cluster',
        palette='dark',
        size='cluster',
        sizes=(100, 300),
        marker='*',  # Use star marker for selected stocks
        edgecolor='red',
        linewidth=1,
        legend=False  # Avoid duplicate legend entries
    )

    plt.xlabel('Average Percent Change (%)')
    plt.ylabel('Volatility (Std of Pct Change)')
    plt.title('Stock Clustering: Avg Pct Change vs Volatility (Selected Stocks Highlighted)')
    plt.legend().remove()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_plot.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Evaluate clustering quality using silhouette score.

    Args:
        X: Scaled feature array.
        labels: Cluster labels for each sample.

    Returns:
        Silhouette score (higher is better, range: [-1, 1]).
    """
    if len(set(labels)) > 1:  # Silhouette score requires at least 2 clusters
        score = silhouette_score(X, labels)
        print(f"Silhouette Score: {score:.3f}")
        return score
    print("Silhouette Score not applicable (single cluster).")
    return np.nan


def cluster_stocks() -> pd.DataFrame:
    """
    Perform clustering on stock data and select promising stocks.

    Returns:
        DataFrame of selected stocks and dictionary of stock data.
    """
    print("Starting clustering process...")
    # Load and prepare data
    stock_data = load_stock_data()
    feature_df, ts_codes = prepare_clustering_data(stock_data)

    if feature_df.empty:
        raise ValueError("No valid features available for clustering.")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    # Determine optimal number of clusters
    optimal_k = plot_dendrogram(X_scaled)

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    clusters = clustering.fit_predict(X_scaled)

    # Evaluate clustering
    silhouette = evaluate_clustering(X_scaled, clusters)

    # Create result DataFrame
    result = pd.DataFrame({
        'ts_code': ts_codes,
        'cluster': clusters,
        'avg_pct_chg': feature_df['avg_pct_chg'],
        'volatility': feature_df['volatility']
    })

    # Calculate cluster statistics
    cluster_stats = result.groupby('cluster').agg({
        'avg_pct_chg': 'mean',
        'volatility': 'mean',
        'ts_code': 'count'
    }).rename(columns={'ts_code': 'count'})

    # Add silhouette score to stats
    cluster_stats['silhouette_score'] = silhouette

    # Score clusters based on normalized metrics
    cluster_stats['norm_pct_chg'] = (
            (cluster_stats['avg_pct_chg'] - cluster_stats['avg_pct_chg'].min()) /
            (cluster_stats['avg_pct_chg'].max() - cluster_stats['avg_pct_chg'].min() + 1e-10)
    )
    cluster_stats['norm_volatility'] = (
            (cluster_stats['volatility'] - cluster_stats['volatility'].min()) /
            (cluster_stats['volatility'].max() - cluster_stats['volatility'].min() + 1e-10)
    )
    cluster_stats['score'] = cluster_stats['norm_pct_chg'] - cluster_stats['norm_volatility']

    # Filter clusters with sufficient size
    valid_clusters = cluster_stats[cluster_stats['count'] >= MIN_CLUSTER_SIZE]
    selected_clusters = valid_clusters.nlargest(2, 'score').index if not valid_clusters.empty else \
        cluster_stats.nlargest(1, 'avg_pct_chg').index

    # Visualize results
    plot_clusters(result, selected_clusters)

    # Save outputs
    selected_stocks = result[result['cluster'].isin(selected_clusters)]

    result.to_csv(os.path.join(OUTPUT_DIR, 'all_clusters.csv'), index=False)
    cluster_stats.to_csv(os.path.join(OUTPUT_DIR, 'cluster_stats.csv'))
    print("Clustering completed successfully.")
    return selected_stocks


selected_stocks = cluster_stocks()

# Print selected stock names
hkex = pd.read_csv(HKEX_FILE)
names = []
for i, row in selected_stocks.iterrows():
    name = hkex[hkex['ts_code'] == row['ts_code']]['name'].iloc[0]
    names.append(name)
selected_stocks['name'] = names
selected_stocks.to_csv(os.path.join(OUTPUT_DIR, 'selected_stocks.csv'), index=False)
