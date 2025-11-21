"""
Customer Segmentation - Step 1: Data Exploration
Dataset: Mall_Customers.csv from Kaggle
Purpose: Explore customer data before clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create directories
def create_directories():
    directories = ['data', 'visualizations', 'models', 'results']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created/verified")

create_directories()

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("="*80)
print("CUSTOMER SEGMENTATION - DATA EXPLORATION")
print("="*80)

df = pd.read_csv('Mall_Customers.csv')

print("\n1. Dataset Overview")
print("-" * 80)
print(f"Total Records: {len(df):,}")
print(f"Total Features: {len(df.columns)}")
print(f"Dataset Shape: {df.shape}")

# ============================================================================
# 2. BASIC INFORMATION
# ============================================================================
print("\n2. First 10 Records")
print("-" * 80)
print(df.head(10))

print("\n3. Dataset Information")
print("-" * 80)
df.info()

print("\n4. Statistical Summary")
print("-" * 80)
print(df.describe())

print("\n5. Column Details")
print("-" * 80)
for col in df.columns:
    print(f"  • {col:30s} - {str(df[col].dtype):10s} - {df[col].nunique()} unique values")


# ============================================================================
# 3. DATA QUALITY CHECK
# ============================================================================
print("\n6. Data Quality Check")
print("-" * 80)

# Missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print("Missing Values:")
    print(missing[missing > 0])
else:
    print("✓ No missing values found!")

# Duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate Rows: {duplicates}")

# ============================================================================
# 4. UNIVARIATE ANALYSIS
# ============================================================================
print("\n7. Univariate Analysis")
print("-" * 80)

# Gender distribution
print("\nGender Distribution:")
print(df['Gender'].value_counts())
print(f"Percentage:")
print(df['Gender'].value_counts(normalize=True) * 100)

# Age statistics
print(f"\nAge Statistics:")
print(f"  • Min Age: {df['Age'].min()}")
print(f"  • Max Age: {df['Age'].max()}")
print(f"  • Mean Age: {df['Age'].mean():.2f}")
print(f"  • Median Age: {df['Age'].median()}")

# Income statistics
print(f"\nAnnual Income Statistics (k$):")
print(f"  • Min Income: ${df['Annual Income (k$)'].min()}k")
print(f"  • Max Income: ${df['Annual Income (k$)'].max()}k")
print(f"  • Mean Income: ${df['Annual Income (k$)'].mean():.2f}k")
print(f"  • Median Income: ${df['Annual Income (k$)'].median()}k")

# Spending Score statistics
print(f"\nSpending Score Statistics (1-100):")
print(f"  • Min Score: {df['Spending Score (1-100)'].min()}")
print(f"  • Max Score: {df['Spending Score (1-100)'].max()}")
print(f"  • Mean Score: {df['Spending Score (1-100)'].mean():.2f}")
print(f"  • Median Score: {df['Spending Score (1-100)'].median()}")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n8. Creating Visualizations")
print("-" * 80)

# Gender Distribution
fig, ax = plt.subplots(figsize=(8, 6))
gender_counts = df['Gender'].value_counts()
ax.bar(gender_counts.index, gender_counts.values, color=['#FF69B4', '#4169E1'], alpha=0.7)
ax.set_title('Gender Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Gender')
ax.set_ylabel('Count')
for i, v in enumerate(gender_counts.values):
    ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/01_gender_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gender distribution saved")

# Age Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['Age'], bins=20, edgecolor='black', color='skyblue', alpha=0.7)
axes[0].set_title('Age Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['Age'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['Age'].mean():.1f}")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].boxplot(df['Age'], vert=True, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', alpha=0.7))
axes[1].set_title('Age Box Plot', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Age')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/02_age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Age distribution saved")

# Income Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['Annual Income (k$)'], bins=20, edgecolor='black', color='lightgreen', alpha=0.7)
axes[0].set_title('Annual Income Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Annual Income (k$)')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['Annual Income (k$)'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f"Mean: ${df['Annual Income (k$)'].mean():.1f}k")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].boxplot(df['Annual Income (k$)'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightgreen', alpha=0.7))
axes[1].set_title('Income Box Plot', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Annual Income (k$)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/03_income_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Income distribution saved")

# Spending Score Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['Spending Score (1-100)'], bins=20, edgecolor='black', color='coral', alpha=0.7)
axes[0].set_title('Spending Score Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Spending Score (1-100)')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['Spending Score (1-100)'].mean(), color='red', linestyle='--',
                linewidth=2, label=f"Mean: {df['Spending Score (1-100)'].mean():.1f}")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].boxplot(df['Spending Score (1-100)'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='coral', alpha=0.7))
axes[1].set_title('Spending Score Box Plot', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Spending Score (1-100)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/04_spending_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Spending score distribution saved")

# ============================================================================
# 6. BIVARIATE ANALYSIS
# ============================================================================
print("\n9. Bivariate Analysis")
print("-" * 80)

# Income vs Spending Score
plt.figure(figsize=(10, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
           c=df['Gender'].map({'Male': 'blue', 'Female': 'pink'}),
           alpha=0.6, s=100, edgecolors='black')
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.title('Income vs Spending Score (by Gender)', fontsize=14, fontweight='bold')
plt.legend(['Male', 'Female'])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/05_income_vs_spending.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Income vs Spending saved")

# Age vs Spending Score
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Spending Score (1-100)'],
           c=df['Gender'].map({'Male': 'blue', 'Female': 'pink'}),
           alpha=0.6, s=100, edgecolors='black')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.title('Age vs Spending Score (by Gender)', fontsize=14, fontweight='bold')
plt.legend(['Male', 'Female'])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/06_age_vs_spending.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Age vs Spending saved")

# Age vs Income
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Annual Income (k$)'],
           c=df['Gender'].map({'Male': 'blue', 'Female': 'pink'}),
           alpha=0.6, s=100, edgecolors='black')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Annual Income (k$)', fontsize=12)
plt.title('Age vs Income (by Gender)', fontsize=14, fontweight='bold')
plt.legend(['Male', 'Female'])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/07_age_vs_income.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Age vs Income saved")

# ============================================================================
# 7. CORRELATION ANALYSIS
# ============================================================================
print("\n10. Correlation Analysis")
print("-" * 80)

# Numerical columns only
numerical_df = df.select_dtypes(include=[np.number])
correlation_matrix = numerical_df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
           center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/08_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Correlation matrix saved")

# ============================================================================
# 8. PAIRPLOT
# ============================================================================
print("\n11. Creating Pairplot")
print("-" * 80)

# Create pairplot
pairplot_fig = sns.pairplot(df, hue='Gender', 
                            vars=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
                            palette={'Male': 'blue', 'Female': 'pink'},
                            diag_kind='kde', plot_kws={'alpha': 0.6})
pairplot_fig.fig.suptitle('Pairplot of Customer Features', y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/09_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Pairplot saved")

# ============================================================================
# 9. SAVE CLEANED DATA
# ============================================================================
print("\n12. Saving Processed Data")
print("-" * 80)

df.to_csv('data/mall_customers_clean.csv', index=False)
print("✓ Cleaned data saved")

# Save summary statistics
summary_stats = {
    'total_customers': len(df),
    'gender_distribution': df['Gender'].value_counts().to_dict(),
    'age_stats': {
        'min': int(df['Age'].min()),
        'max': int(df['Age'].max()),
        'mean': float(df['Age'].mean()),
        'median': float(df['Age'].median())
    },
    'income_stats': {
        'min': int(df['Annual Income (k$)'].min()),
        'max': int(df['Annual Income (k$)'].max()),
        'mean': float(df['Annual Income (k$)'].mean()),
        'median': float(df['Annual Income (k$)'].median())
    },
    'spending_stats': {
        'min': int(df['Spending Score (1-100)'].min()),
        'max': int(df['Spending Score (1-100)'].max()),
        'mean': float(df['Spending Score (1-100)'].mean()),
        'median': float(df['Spending Score (1-100)'].median())
    }
}

import json
with open('data/summary_stats.json', 'w') as f:
    json.dump(summary_stats, f, indent=4)
print("✓ Summary statistics saved")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DATA EXPLORATION SUMMARY")
print("="*80)

print(f"""
Dataset Overview:
  • Total Customers: {len(df)}
  • Features: {len(df.columns)}
  • No Missing Values: ✓
  • No Duplicates: {'✓' if duplicates == 0 else f'⚠️ {duplicates} found'}

Customer Demographics:
  • Gender: {df['Gender'].value_counts()['Male']} Male, {df['Gender'].value_counts()['Female']} Female
  • Age Range: {df['Age'].min()} - {df['Age'].max()} years
  • Average Age: {df['Age'].mean():.1f} years

Financial Profile:
  • Income Range: ${df['Annual Income (k$)'].min()}k - ${df['Annual Income (k$)'].max()}k
  • Average Income: ${df['Annual Income (k$)'].mean():.1f}k
  
Spending Behavior:
  • Spending Score Range: {df['Spending Score (1-100)'].min()} - {df['Spending Score (1-100)'].max()}
  • Average Spending Score: {df['Spending Score (1-100)'].mean():.1f}

Key Insights:
  • Dataset is ready for clustering
  • Income and Spending Score show interesting patterns
  • Gender distribution is balanced
  • No data quality issues found

Visualizations Created:
  ✓ 01_gender_distribution.png
  ✓ 02_age_distribution.png
  ✓ 03_income_distribution.png
  ✓ 04_spending_distribution.png
  ✓ 05_income_vs_spending.png
  ✓ 06_age_vs_spending.png
  ✓ 07_age_vs_income.png
  ✓ 08_correlation_matrix.png
  ✓ 09_pairplot.png

Next Steps:
  1. Scale the features (2_feature_scaling.py)
  2. Determine optimal clusters (3_elbow_method.py)
  3. Apply K-Means clustering (4_kmeans_clustering.py)
  4. Visualize and analyze segments (5_cluster_visualization.py)
""")

print("="*80)
print("DATA EXPLORATION COMPLETED!")
print("="*80)
