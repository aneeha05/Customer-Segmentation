# 🛍️ Customer Segmentation with K-Means Clustering

A complete machine learning project for customer segmentation using K-Means clustering on Mall Customers dataset from Kaggle.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange?logo=scikit-learn)

## 📊 Project Overview

This project segments mall customers into distinct groups based on their:
- **Annual Income (k$)**
- **Spending Score (1-100)**

Using **K-Means clustering**, we identify customer segments for targeted marketing strategies.

## 📁 Project Structure

```
customer-segmentation/
│
├── 1_data_exploration.py          # Step 1: EDA
├── 2_feature_scaling.py            # Step 2: Feature scaling
├── 3_kmeans_clustering.py          # Step 3: Elbow method + K-Means
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Dependencies
├── README.md                       # This file
│
├── data/
│   ├── mall_customers_clean.csv
│   ├── features_scaled_primary.csv
│   ├── customers_with_clusters.csv
│   └── summary_stats.json
│
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler_primary.pkl
│   └── cluster_info.json
│
├── visualizations/
│   ├── 01-09_*.png                 # EDA plots
│   ├── 10-12_*.png                 # Scaling plots
│   └── 13-16_*.png                 # Clustering plots
│
└── results/
    └── cluster_profiles.csv
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download `Mall_Customers.csv` from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) and place it in the project root.

### 3. Run Analysis Scripts

```bash
# Step 1: Data Exploration
python 1_data_exploration.py

# Step 2: Feature Scaling
python 2_feature_scaling.py

# Step 3: K-Means Clustering (includes Elbow Method)
python 3_kmeans_clustering.py
```

### 4. Launch Dashboard

```bash
streamlit run app.py
```

Dashboard will open at `http://localhost:8501`

## 📊 Dataset Information

**Source**: Mall Customers Dataset (Kaggle)  
**Records**: 200 customers  
**Features**:
- `CustomerID`: Unique identifier
- `Gender`: Male/Female
- `Age`: Customer age (18-70)
- `Annual Income (k$)`: Income in thousands (15-140)
- `Spending Score (1-100)`: Shopping behavior score

## 🎯 Methodology

### 1. **Data Exploration** (`1_data_exploration.py`)
- Load and analyze dataset
- Check for missing values/duplicates
- Create univariate and bivariate visualizations
- Analyze correlations
- **Output**: 9 visualizations

### 2. **Feature Scaling** (`2_feature_scaling.py`)
- Select features for clustering (Income + Spending)
- Apply **StandardScaler** (mean=0, std=1)
- Visualize before/after scaling
- **Why**: K-Means is distance-based and sensitive to scale
- **Output**: 3 visualizations, scaled features

### 3. **K-Means Clustering** (`3_kmeans_clustering.py`)

#### Elbow Method
- Test K=2 to K=10
- Calculate **Inertia** (within-cluster sum of squares)
- Calculate **Silhouette Score** (cluster quality)
- Calculate **Davies-Bouldin Score** (cluster separation)
- **Optimal K**: Determined by elbow point (typically K=5)

#### Clustering
- Apply K-Means with optimal K
- Assign cluster labels
- Analyze cluster centroids
- **Output**: 4 visualizations, cluster assignments

## 📈 Results

### Typical Segments (K=5)

| Cluster | Income | Spending | Profile |
|---------|--------|----------|---------|
| **0** | High | High | 💎 **Premium Customers** - VIP treatment |
| **1** | High | Low | 💰 **Careful Spenders** - Need promotions |
| **2** | Low | High | 🎯 **Impulsive Buyers** - Budget options |
| **3** | Low | Low | 🏷️ **Budget Conscious** - Discount focus |
| **4** | Medium | Medium | ⚖️ **Average Customers** - Balanced approach |

### Performance Metrics
- **Silhouette Score**: ~0.45-0.55 (Good separation)
- **Inertia**: Minimized at optimal K
- **Davies-Bouldin Score**: <1.5 (Well-separated clusters)

## 🖥️ Streamlit Dashboard Features

### 🏠 **Overview**
- Key metrics (Total customers, segments, avg income/spending)
- Dataset preview
- Quick statistics

### 📊 **Data Analysis**
- Feature distributions (Age, Income, Spending)
- Correlation heatmap
- Scatter plots (Income vs Spending, Age vs Spending)
- Gender comparisons

### 🎯 **Clustering Results**
- 2D cluster visualization (Income vs Spending)
- 3D cluster visualization (Age + Income + Spending)
- Cluster metrics
- Interactive plots with hover details

### 👥 **Segment Profiles**
- Detailed statistics for each cluster
- Segment sizes and characteristics
- Marketing recommendations
- Gender distribution per segment

### 🔮 **Predict Segment**
- Interactive form (Age, Income, Gender, Spending)
- Real-time prediction
- Visual positioning in segments
- Segment characteristics

## 📊 Visualizations Created

### EDA (9 plots)
1. Gender distribution
2. Age distribution (histogram + boxplot)
3. Income distribution (histogram + boxplot)
4. Spending distribution (histogram + boxplot)
5. Income vs Spending (colored by gender)
6. Age vs Spending
7. Age vs Income
8. Correlation matrix
9. Pairplot

### Scaling (3 plots)
10. Before scaling visualization
11. After scaling visualization
12. Distribution comparison

### Clustering (4 plots)
13. Elbow method (3 metrics)
14. Clusters on scaled features
15. Clusters on original scale
16. Cluster profiles comparison

## 🎯 Business Applications

### Marketing Strategies

**💎 Premium Customers (High Income, High Spending)**
- VIP programs and exclusive offers
- Premium product lines
- Personalized service
- Loyalty rewards

**💰 Careful Spenders (High Income, Low Spending)**
- Targeted promotions
- Quality assurance messaging
- Value propositions
- Membership benefits

**🎯 Impulsive Buyers (Low Income, High Spending)**
- Budget-friendly payment plans
- Installment options
- Flash sales
- Limited-time offers

**🏷️ Budget Conscious (Low Income, Low Spending)**
- Discount programs
- Clearance sales
- Bundle deals
- Cost-saving tips

**⚖️ Average Customers (Medium Income, Medium Spending)**
- Standard promotions
- Seasonal offers
- Loyalty programs
- Regular engagement

## 🔬 Technical Details

### K-Means Algorithm
```python
from sklearn.cluster import KMeans

# Initialize
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

# Fit
kmeans.fit(X_scaled)

# Predict
clusters = kmeans.predict(X_scaled)
```

### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

# Initialize
scaler = StandardScaler()

# Fit and transform
X_scaled = scaler.fit_transform(X)

# Result: mean=0, std=1 for all features
```

### Evaluation Metrics

**Inertia (WCSS)**
- Sum of squared distances to nearest centroid
- Lower is better
- Used in Elbow Method

**Silhouette Score**
- Measures cluster cohesion and separation
- Range: -1 to +1
- >0.5 is good

**Davies-Bouldin Score**
- Average similarity between clusters
- Lower is better
- <1.0 is excellent

## 🛠️ Customization

### Change Number of Clusters

Edit `3_kmeans_clustering.py`:
```python
optimal_k = 6  # Change from 5 to 6
```

### Use Different Features

Edit `2_feature_scaling.py`:
```python
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
```

### Adjust Dashboard Theme

Edit `app.py`:
```python
st.set_page_config(
    page_title="Your Title",
    page_icon="🎨",
    layout="wide"
)
```

## 📊 Key Insights

1. **Income and Spending are not strongly correlated**
   - High earners don't always spend more
   - Spending behavior varies across income levels

2. **5 distinct customer segments** identified
   - Each requires different marketing approach
   - Clear separation in 2D space

3. **Gender distribution is balanced** across segments
   - No significant gender bias
   - Strategies should be gender-neutral

4. **Age influences spending patterns**
   - Younger customers tend to spend more
   - Consider age in targeted campaigns

## 🚀 Advanced Usage

### Batch Prediction

```python
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('models/kmeans_model.pkl')
scaler = joblib.load('models/scaler_primary.pkl')

# New customers
new_customers = pd.DataFrame({
    'Annual Income (k$)': [50, 75, 30],
    'Spending Score (1-100)': [60, 80, 40]
})

# Scale and predict
X_scaled = scaler.transform(new_customers)
clusters = model.predict(X_scaled)

print(f"Predicted clusters: {clusters}")
```

### Export Segments

```python
df = pd.read_csv('data/customers_with_clusters.csv')

for cluster in df['Cluster'].unique():
    segment_df = df[df['Cluster'] == cluster]
    segment_df.to_csv(f'segment_{cluster}.csv', index=False)
```

## 📝 Best Practices

1. **Always scale features** for K-Means
2. **Use Elbow Method** to find optimal K
3. **Validate with multiple metrics** (Silhouette, Davies-Bouldin)
4. **Interpret clusters** in business context
5. **Update regularly** with new customer data
6. **A/B test** marketing strategies per segment

## 🐛 Troubleshooting

**Issue**: Low Silhouette Score (<0.3)
- Try different K values
- Check feature scaling
- Consider different features

**Issue**: No clear elbow
- Use Silhouette Score as primary metric
- Try K-Means++ initialization
- Consider hierarchical clustering

**Issue**: Unbalanced clusters
- Normal for real-world data
- Focus on cluster quality, not size
- Use stratified sampling if needed

## 📚 Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: K-Means, scaling, metrics
- `matplotlib`: Static visualizations
- `seaborn`: Statistical plots
- `plotly`: Interactive visualizations
- `streamlit`: Web dashboard
- `joblib`: Model serialization

## 🎓 Learning Resources

- [K-Means Clustering Explained](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering))
- [Silhouette Analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
- [Feature Scaling](https://scikit-learn.org/stable/modules/preprocessing.html)

## 📄 License

MIT License - Free for educational and commercial use

## 👥 Author

Data Science Team

## 🎉 Acknowledgments

- Kaggle for the Mall Customers dataset
- Scikit-learn for ML algorithms
- Streamlit for the dashboard framework

---

**Ready to segment your customers! 🚀**
