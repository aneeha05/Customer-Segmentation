"""
Customer Segmentation - Step 2: Feature Scaling
Purpose: Scale features for K-Means clustering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("="*80)
print("CUSTOMER SEGMENTATION - FEATURE SCALING")
print("="*80)

df = pd.read_csv('data/mall_customers_clean.csv')

print(f"\nLoaded {len(df)} customer records")
print(f"Features: {list(df.columns)}")

# ============================================================================
# 2. SELECT FEATURES FOR CLUSTERING
# ============================================================================
print("\n1. Feature Selection for Clustering")
print("-" * 80)

# We'll use Income and Spending Score (as specified)
# But also prepare Age-based clustering for comparison
features_income_spending = ['Annual Income (k$)', 'Spending Score (1-100)']
features_all = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

print(f"\nPrimary Features (Income + Spending): {features_income_spending}")
print(f"Extended Features (Age + Income + Spending): {features_all}")

# Extract feature sets
X_income_spending = df[features_income_spending].values
X_all = df[features_all].values

print(f"\nPrimary Feature Matrix Shape: {X_income_spending.shape}")
print(f"Extended Feature Matrix Shape: {X_all.shape}")

# ============================================================================
# 3. VISUALIZE BEFORE SCALING
# ============================================================================
print("\n2. Visualizing Features Before Scaling")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before scaling - Income vs Spending
axes[0].scatter(X_income_spending[:, 0], X_income_spending[:, 1], 
               alpha=0.6, s=100, c='blue', edgecolors='black')
axes[0].set_xlabel('Annual Income (k$)', fontsize=12)
axes[0].set_ylabel('Spending Score (1-100)', fontsize=12)
axes[0].set_title('Before Scaling', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

# Show the scale difference
axes[0].text(0.05, 0.95, f'Income range: {X_income_spending[:, 0].min():.0f} - {X_income_spending[:, 0].max():.0f}',
            transform=axes[0].transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[0].text(0.05, 0.85, f'Spending range: {X_income_spending[:, 1].min():.0f} - {X_income_spending[:, 1].max():.0f}',
            transform=axes[0].transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Statistics
axes[1].axis('off')
stats_text = f"""
Feature Statistics (Before Scaling)

Annual Income (k$):
  • Min: {df['Annual Income (k$)'].min():.2f}
  • Max: {df['Annual Income (k$)'].max():.2f}
  • Mean: {df['Annual Income (k$)'].mean():.2f}
  • Std: {df['Annual Income (k$)'].std():.2f}

Spending Score (1-100):
  • Min: {df['Spending Score (1-100)'].min():.2f}
  • Max: {df['Spending Score (1-100)'].max():.2f}
  • Mean: {df['Spending Score (1-100)'].mean():.2f}
  • Std: {df['Spending Score (1-100)'].std():.2f}

⚠️ Different scales can bias K-Means!
StandardScaler will normalize both features.
"""
axes[1].text(0.1, 0.9, stats_text, transform=axes[1].transAxes,
            fontsize=11, verticalalignment='top', family='monospace')

plt.tight_layout()
plt.savefig('visualizations/10_before_scaling.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Before scaling visualization saved")

# ============================================================================
# 4. APPLY STANDARD SCALING
# ============================================================================
print("\n3. Applying StandardScaler")
print("-" * 80)

# Scale primary features (Income + Spending)
scaler_primary = StandardScaler()
X_scaled_primary = scaler_primary.fit_transform(X_income_spending)

print("\n✓ Primary features scaled")
print(f"   Mean: {X_scaled_primary.mean(axis=0)}")
print(f"   Std: {X_scaled_primary.std(axis=0)}")

# Scale all features (Age + Income + Spending)
scaler_all = StandardScaler()
X_scaled_all = scaler_all.fit_transform(X_all)

print("\n✓ All features scaled")
print(f"   Mean: {X_scaled_all.mean(axis=0)}")
print(f"   Std: {X_scaled_all.std(axis=0)}")

# ============================================================================
# 5. VISUALIZE AFTER SCALING
# ============================================================================
print("\n4. Visualizing Features After Scaling")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# After scaling - Income vs Spending
axes[0].scatter(X_scaled_primary[:, 0], X_scaled_primary[:, 1],
               alpha=0.6, s=100, c='green', edgecolors='black')
axes[0].set_xlabel('Annual Income (Scaled)', fontsize=12)
axes[0].set_ylabel('Spending Score (Scaled)', fontsize=12)
axes[0].set_title('After Scaling', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)

# Comparison
axes[1].scatter(X_income_spending[:, 0], X_income_spending[:, 1],
               alpha=0.4, s=80, c='blue', edgecolors='black', label='Before')
axes[1].scatter(X_scaled_primary[:, 0] * 30 + 60, X_scaled_primary[:, 1] * 20 + 50,
               alpha=0.4, s=80, c='green', edgecolors='black', label='After (rescaled for comparison)')
axes[1].set_xlabel('Feature 1', fontsize=12)
axes[1].set_ylabel('Feature 2', fontsize=12)
axes[1].set_title('Comparison (Illustrative)', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/11_after_scaling.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ After scaling visualization saved")

# ============================================================================
# 6. DISTRIBUTION COMPARISON
# ============================================================================
print("\n5. Comparing Distributions")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Income before scaling
axes[0, 0].hist(X_income_spending[:, 0], bins=20, edgecolor='black', 
               alpha=0.7, color='lightblue')
axes[0, 0].set_title('Annual Income - Before Scaling', fontweight='bold')
axes[0, 0].set_xlabel('Annual Income (k$)')
axes[0, 0].set_ylabel('Frequency')

# Income after scaling
axes[0, 1].hist(X_scaled_primary[:, 0], bins=20, edgecolor='black',
               alpha=0.7, color='lightgreen')
axes[0, 1].set_title('Annual Income - After Scaling', fontweight='bold')
axes[0, 1].set_xlabel('Scaled Annual Income')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(x=0, color='r', linestyle='--', label='Mean = 0')
axes[0, 1].legend()

# Spending before scaling
axes[1, 0].hist(X_income_spending[:, 1], bins=20, edgecolor='black',
               alpha=0.7, color='lightcoral')
axes[1, 0].set_title('Spending Score - Before Scaling', fontweight='bold')
axes[1, 0].set_xlabel('Spending Score (1-100)')
axes[1, 0].set_ylabel('Frequency')

# Spending after scaling
axes[1, 1].hist(X_scaled_primary[:, 1], bins=20, edgecolor='black',
               alpha=0.7, color='lightyellow')
axes[1, 1].set_title('Spending Score - After Scaling', fontweight='bold')
axes[1, 1].set_xlabel('Scaled Spending Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].axvline(x=0, color='r', linestyle='--', label='Mean = 0')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('visualizations/12_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Distribution comparison saved")

# ============================================================================
# 7. SAVE SCALED DATA
# ============================================================================
print("\n6. Saving Scaled Data")
print("-" * 80)

# Save primary scaled features (Income + Spending)
scaled_df_primary = pd.DataFrame(
    X_scaled_primary,
    columns=['Annual_Income_Scaled', 'Spending_Score_Scaled']
)
scaled_df_primary.to_csv('data/features_scaled_primary.csv', index=False)
print("✓ Primary scaled features saved")

# Save all scaled features (Age + Income + Spending)
scaled_df_all = pd.DataFrame(
    X_scaled_all,
    columns=['Age_Scaled', 'Annual_Income_Scaled', 'Spending_Score_Scaled']
)
scaled_df_all.to_csv('data/features_scaled_all.csv', index=False)
print("✓ All scaled features saved")

# Save original data with scaled features combined
df_combined = df.copy()
df_combined['Annual_Income_Scaled'] = X_scaled_primary[:, 0]
df_combined['Spending_Score_Scaled'] = X_scaled_primary[:, 1]
df_combined.to_csv('data/customers_with_scaled_features.csv', index=False)
print("✓ Combined dataset saved")

# ============================================================================
# 8. SAVE SCALERS
# ============================================================================
print("\n7. Saving Scaler Objects")
print("-" * 80)

joblib.dump(scaler_primary, 'models/scaler_primary.pkl')
print("✓ Primary scaler saved")

joblib.dump(scaler_all, 'models/scaler_all.pkl')
print("✓ All features scaler saved")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FEATURE SCALING SUMMARY")
print("="*80)

print(f"""
Scaling Method: StandardScaler
  • Transforms features to have mean=0 and std=1
  • Essential for distance-based algorithms like K-Means

Primary Features (for clustering):
  • Annual Income (k$)
  • Spending Score (1-100)
  • Shape: {X_scaled_primary.shape}

Scaled Statistics:
  • Mean: ~0 for both features
  • Standard Deviation: ~1 for both features
  • Range: approximately -3 to +3

Why Scaling Matters for K-Means:
  ✓ Equal weight to all features
  ✓ Prevents bias from different scales
  ✓ Improves clustering quality
  ✓ Faster convergence

Files Saved:
  ✓ features_scaled_primary.csv (Income + Spending)
  ✓ features_scaled_all.csv (Age + Income + Spending)
  ✓ customers_with_scaled_features.csv (Original + Scaled)
  ✓ scaler_primary.pkl (Scaler object)
  ✓ scaler_all.pkl (Scaler object)

Visualizations Created:
  ✓ 10_before_scaling.png
  ✓ 11_after_scaling.png
  ✓ 12_distribution_comparison.png

Next Steps:
  1. Determine optimal number of clusters (3_elbow_method.py)
  2. Apply K-Means clustering
  3. Analyze and visualize segments
""")

print("="*80)
print("FEATURE SCALING COMPLETED!")
print("="*80)
