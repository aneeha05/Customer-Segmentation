"""
Customer Segmentation - Step 3: Elbow Method + K-Means Clustering
Purpose: Find optimal clusters and apply K-Means
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CUSTOMER SEGMENTATION - K-MEANS CLUSTERING")
print("="*80)

# Load scaled data
X_scaled = pd.read_csv('data/features_scaled_primary.csv').values
df_original = pd.read_csv('data/mall_customers_clean.csv')

print(f"\nLoaded {len(X_scaled)} customers")
print(f"Features: Annual Income + Spending Score (scaled)")

# ============================================================================
# 1. ELBOW METHOD
# ============================================================================
print("\n1. Elbow Method - Finding Optimal Clusters")
print("-" * 80)

inertias = []
silhouette_scores = []
davies_bouldin_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    davies_bouldin_scores.append(davies_bouldin_score(X_scaled, kmeans.labels_))
    
    print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")

# Plot Elbow Method
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Elbow curve (Inertia)
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].axvline(x=5, color='red', linestyle='--', label='Optimal K=5')
axes[0].legend()

# Silhouette Score
axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score (Higher is Better)', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

# Davies-Bouldin Score
axes[2].plot(K_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
axes[2].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[2].set_ylabel('Davies-Bouldin Score', fontsize=12)
axes[2].set_title('Davies-Bouldin Score (Lower is Better)', fontsize=14, fontweight='bold')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/13_elbow_method.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Elbow method visualization saved")

# ============================================================================
# 2. APPLY K-MEANS WITH OPTIMAL K
# ============================================================================
optimal_k = 5  # Based on elbow method and domain knowledge
print(f"\n2. Applying K-Means with K={optimal_k}")
print("-" * 80)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

print(f"✓ Clustering completed")
print(f"  Inertia: {kmeans.inertia_:.2f}")
print(f"  Silhouette Score: {silhouette_score(X_scaled, clusters):.3f}")

# Add cluster labels to original data
df_original['Cluster'] = clusters
df_original.to_csv('data/customers_with_clusters.csv', index=False)
print("✓ Cluster labels added to dataset")

# ============================================================================
# 3. CLUSTER ANALYSIS
# ============================================================================
print("\n3. Cluster Analysis")
print("-" * 80)

print("\nCluster Size Distribution:")
cluster_counts = pd.Series(clusters).value_counts().sort_index()
for cluster, count in cluster_counts.items():
    print(f"  Cluster {cluster}: {count} customers ({count/len(clusters)*100:.1f}%)")

print("\nCluster Centroids (Scaled):")
centroids = kmeans.cluster_centers_
for i, centroid in enumerate(centroids):
    print(f"  Cluster {i}: Income={centroid[0]:.3f}, Spending={centroid[1]:.3f}")

# ============================================================================
# 4. VISUALIZE CLUSTERS
# ============================================================================
print("\n4. Creating Cluster Visualizations")
print("-" * 80)

# 2D Scatter Plot with Clusters
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black']

for i in range(optimal_k):
    cluster_points = X_scaled[clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
               s=100, c=colors[i], label=f'Cluster {i}', alpha=0.6, edgecolors='black')

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1],
           s=300, c='gold', marker='*', edgecolors='black', linewidths=2,
           label='Centroids', zorder=10)

plt.xlabel('Annual Income (Scaled)', fontsize=12, fontweight='bold')
plt.ylabel('Spending Score (Scaled)', fontsize=12, fontweight='bold')
plt.title(f'K-Means Clustering (K={optimal_k})', fontsize=16, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/14_clusters_scaled.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Scaled clusters visualization saved")

# Original Scale Visualization
plt.figure(figsize=(12, 8))

for i in range(optimal_k):
    cluster_mask = clusters == i
    plt.scatter(df_original[cluster_mask]['Annual Income (k$)'],
               df_original[cluster_mask]['Spending Score (1-100)'],
               s=100, c=colors[i], label=f'Cluster {i}', alpha=0.6, edgecolors='black')

plt.xlabel('Annual Income (k$)', fontsize=12, fontweight='bold')
plt.ylabel('Spending Score (1-100)', fontsize=12, fontweight='bold')
plt.title(f'Customer Segments (Original Scale)', fontsize=16, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/15_clusters_original.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Original scale clusters visualization saved")

# ============================================================================
# 5. CLUSTER PROFILES
# ============================================================================
print("\n5. Analyzing Cluster Profiles")
print("-" * 80)

cluster_profiles = df_original.groupby('Cluster').agg({
    'Age': ['mean', 'min', 'max'],
    'Annual Income (k$)': ['mean', 'min', 'max'],
    'Spending Score (1-100)': ['mean', 'min', 'max']
}).round(2)

print("\nCluster Profiles:")
print(cluster_profiles)

# Save profiles
cluster_profiles.to_csv('results/cluster_profiles.csv')
print("\n✓ Cluster profiles saved")

# Visualize profiles
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for idx, feature in enumerate(features):
    cluster_means = df_original.groupby('Cluster')[feature].mean()
    axes[idx].bar(range(optimal_k), cluster_means, color=colors[:optimal_k], alpha=0.7, edgecolor='black')
    axes[idx].set_xlabel('Cluster', fontsize=11)
    axes[idx].set_ylabel(f'Mean {feature}', fontsize=11)
    axes[idx].set_title(f'Mean {feature} by Cluster', fontsize=12, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(cluster_means):
        axes[idx].text(i, v + max(cluster_means)*0.02, f'{v:.1f}', 
                      ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/16_cluster_profiles.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Cluster profiles visualization saved")

# ============================================================================
# 6. SAVE MODEL
# ============================================================================
print("\n6. Saving Model")
print("-" * 80)

joblib.dump(kmeans, 'models/kmeans_model.pkl')
print("✓ K-Means model saved")

# Save cluster info
cluster_info = {
    'n_clusters': optimal_k,
    'inertia': float(kmeans.inertia_),
    'silhouette_score': float(silhouette_score(X_scaled, clusters)),
    'cluster_sizes': cluster_counts.to_dict(),
    'centroids': centroids.tolist()
}

import json
with open('models/cluster_info.json', 'w') as f:
    json.dump(cluster_info, f, indent=4)
print("✓ Cluster information saved")

# ============================================================================
# 7. SEGMENT INTERPRETATION
# ============================================================================
print("\n" + "="*80)
print("CLUSTERING RESULTS SUMMARY")
print("="*80)

print(f"""
Optimal Number of Clusters: {optimal_k}

Model Performance:
  • Inertia: {kmeans.inertia_:.2f}
  • Silhouette Score: {silhouette_score(X_scaled, clusters):.3f}

Cluster Segments (Typical Interpretation):
  
  Cluster 0: {cluster_counts[0]} customers
    → Analyze profile in cluster_profiles.csv
  
  Cluster 1: {cluster_counts[1]} customers
    → Analyze profile in cluster_profiles.csv
  
  Cluster 2: {cluster_counts[2]} customers
    → Analyze profile in cluster_profiles.csv
  
  Cluster 3: {cluster_counts[3]} customers
    → Analyze profile in cluster_profiles.csv
  
  Cluster 4: {cluster_counts[4]} customers
    → Analyze profile in cluster_profiles.csv

Common Segment Names (based on Income vs Spending):
  • High Income, High Spending → "Premium Customers"
  • High Income, Low Spending → "Careful Spenders"
  • Low Income, High Spending → "Impulsive Buyers"
  • Low Income, Low Spending → "Budget Conscious"
  • Medium Income, Medium Spending → "Average Customers"

Files Saved:
  ✓ customers_with_clusters.csv
  ✓ cluster_profiles.csv
  ✓ kmeans_model.pkl
  ✓ cluster_info.json

Visualizations Created:
  ✓ 13_elbow_method.png
  ✓ 14_clusters_scaled.png
  ✓ 15_clusters_original.png
  ✓ 16_cluster_profiles.png

Next Step:
  • Run Streamlit dashboard: streamlit run app.py
""")

print("="*80)
print("K-MEANS CLUSTERING COMPLETED!")
print("="*80)
