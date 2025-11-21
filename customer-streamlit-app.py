"""
Customer Segmentation Dashboard
Streamlit Application for Mall Customers Clustering
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .cluster-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/customers_with_clusters.csv')
        return df
    except:
        st.error("Data file not found. Please run clustering scripts first.")
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/kmeans_model.pkl')
        return model
    except:
        return None

@st.cache_data
def load_cluster_info():
    try:
        with open('models/cluster_info.json', 'r') as f:
            return json.load(f)
    except:
        return None

# Sidebar
st.sidebar.markdown("# 🛍️ Customer Segmentation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 Data Analysis", "🎯 Clustering Results", "👥 Segment Profiles", "🔮 Predict Segment"]
)

st.sidebar.markdown("---")
st.sidebar.info("**K-Means Clustering** on Mall Customers based on Income and Spending Score")

# Load data
df = load_data()
model = load_model()
cluster_info = load_cluster_info()

if df is None:
    st.stop()

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Overview":
    st.markdown('<h1 class="main-header">🛍️ Customer Segmentation Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h3>🎯 Project Overview</h3>
    <p>This dashboard presents customer segmentation analysis using <strong>K-Means clustering</strong> 
    on mall customer data. Customers are grouped based on their <strong>Annual Income</strong> and 
    <strong>Spending Score</strong> to identify distinct market segments.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color: white;">👥 {len(df)}</h3>
            <p style="margin:5px 0 0 0; color: rgba(255,255,255,0.9);">Total Customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        n_clusters = df['Cluster'].nunique()
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3 style="margin:0; color: white;">🎯 {n_clusters}</h3>
            <p style="margin:5px 0 0 0; color: rgba(255,255,255,0.9);">Segments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_income = df['Annual Income (k$)'].mean()
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3 style="margin:0; color: white;">${avg_income:.1f}k</h3>
            <p style="margin:5px 0 0 0; color: rgba(255,255,255,0.9);">Avg Income</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_spending = df['Spending Score (1-100)'].mean()
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h3 style="margin:0; color: white;">{avg_spending:.1f}</h3>
            <p style="margin:5px 0 0 0; color: rgba(255,255,255,0.9);">Avg Spending Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dataset Preview
    st.markdown("### 📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Quick Stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Gender Distribution")
        gender_counts = df['Gender'].value_counts()
        fig = px.pie(values=gender_counts.values, names=gender_counts.index,
                    color_discrete_sequence=['#FF69B4', '#4169E1'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Cluster Distribution")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                    labels={'x': 'Cluster', 'y': 'Count'},
                    color=cluster_counts.values,
                    color_continuous_scale='Viridis')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# DATA ANALYSIS PAGE
# ============================================================================
elif page == "📊 Data Analysis":
    st.markdown("# 📊 Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlations", "📦 Comparisons"])
    
    with tab1:
        st.markdown("### Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='Age', nbins=20,
                             title='Age Distribution',
                             color_discrete_sequence=['steelblue'])
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.histogram(df, x='Annual Income (k$)', nbins=20,
                             title='Income Distribution',
                             color_discrete_sequence=['lightgreen'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='Spending Score (1-100)', nbins=20,
                             title='Spending Score Distribution',
                             color_discrete_sequence=['coral'])
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.box(df, y='Annual Income (k$)', x='Gender',
                        title='Income by Gender',
                        color='Gender',
                        color_discrete_map={'Male': '#4169E1', 'Female': '#FF69B4'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Correlation Analysis")
        
        numerical_df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
        corr_matrix = numerical_df.corr()
        
        fig = px.imshow(corr_matrix, 
                       labels=dict(color="Correlation"),
                       x=corr_matrix.columns,
                       y=corr_matrix.columns,
                       color_continuous_scale='RdBu_r',
                       aspect="auto",
                       text_auto='.2f')
        fig.update_layout(title="Correlation Heatmap", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Feature Comparisons")
        
        fig = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)',
                        color='Gender', size='Age',
                        title='Income vs Spending Score',
                        color_discrete_map={'Male': '#4169E1', 'Female': '#FF69B4'})
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.scatter(df, x='Age', y='Spending Score (1-100)',
                        color='Gender',
                        title='Age vs Spending Score',
                        color_discrete_map={'Male': '#4169E1', 'Female': '#FF69B4'})
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# CLUSTERING RESULTS PAGE
# ============================================================================
elif page == "🎯 Clustering Results":
    st.markdown("# 🎯 Clustering Results")
    
    if cluster_info:
        col1, col2, col3 = st.columns(3)
        col1.metric("Clusters", cluster_info['n_clusters'])
        col2.metric("Silhouette Score", f"{cluster_info['silhouette_score']:.3f}")
        col3.metric("Inertia", f"{cluster_info['inertia']:.2f}")
    
    st.markdown("### 🗺️ Customer Segments Visualization")
    
    # 2D Cluster Plot
    fig = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)',
                    color='Cluster', 
                    title='Customer Segments (K-Means Clustering)',
                    color_continuous_scale='Viridis',
                    size='Age',
                    hover_data=['CustomerID', 'Gender', 'Age'])
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # 3D Visualization
    st.markdown("### 🎲 3D Cluster Visualization")
    fig = px.scatter_3d(df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
                       color='Cluster',
                       title='3D Customer Segmentation',
                       color_continuous_scale='Viridis',
                       hover_data=['CustomerID', 'Gender'])
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SEGMENT PROFILES PAGE
# ============================================================================
elif page == "👥 Segment Profiles":
    st.markdown("# 👥 Detailed Segment Profiles")
    
    # Cluster statistics
    cluster_stats = df.groupby('Cluster').agg({
        'Age': 'mean',
        'Annual Income (k$)': 'mean',
        'Spending Score (1-100)': 'mean',
        'CustomerID': 'count'
    }).round(2)
    cluster_stats.columns = ['Avg Age', 'Avg Income', 'Avg Spending', 'Size']
    
    st.markdown("### 📊 Cluster Statistics")
    st.dataframe(cluster_stats, use_container_width=True)
    
    # Individual cluster profiles
    for cluster in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == cluster]
        
        st.markdown(f"""
        <div class="cluster-card">
        <h3>🎯 Cluster {cluster} - {len(cluster_df)} Customers</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Age", f"{cluster_df['Age'].mean():.1f}")
        col2.metric("Avg Income", f"${cluster_df['Annual Income (k$)'].mean():.1f}k")
        col3.metric("Avg Spending", f"{cluster_df['Spending Score (1-100)'].mean():.1f}")
        col4.metric("Gender Split", f"M:{(cluster_df['Gender']=='Male').sum()} F:{(cluster_df['Gender']=='Female').sum()}")
        
        # Characteristics
        avg_income = cluster_df['Annual Income (k$)'].mean()
        avg_spending = cluster_df['Spending Score (1-100)'].mean()
        
        if avg_income > 70 and avg_spending > 60:
            segment_name = "💎 Premium Customers"
            description = "High income, high spending - VIP treatment recommended"
        elif avg_income > 70 and avg_spending < 40:
            segment_name = "💰 Careful Spenders"
            description = "High income, low spending - need targeted promotions"
        elif avg_income < 40 and avg_spending > 60:
            segment_name = "🎯 Impulsive Buyers"
            description = "Low income, high spending - offer budget-friendly options"
        elif avg_income < 40 and avg_spending < 40:
            segment_name = "🏷️ Budget Conscious"
            description = "Low income, low spending - focus on discounts"
        else:
            segment_name = "⚖️ Average Customers"
            description = "Medium income, medium spending - balanced approach"
        
        st.info(f"**{segment_name}**: {description}")

# ============================================================================
# PREDICTION PAGE
# ============================================================================
elif page == "🔮 Predict Segment":
    st.markdown("# 🔮 Predict Customer Segment")
    
    st.markdown("""
    <div style="background: #f8f9fa; padding: 15px; border-radius: 10px;">
    Enter customer details to predict which segment they belong to.
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 70, 30)
            income = st.slider("Annual Income (k$)", 15, 140, 60)
        
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            spending = st.slider("Spending Score (1-100)", 1, 100, 50)
        
        submitted = st.form_submit_button("🔮 Predict Segment")
    
    if submitted and model is not None:
        # Prepare input
        from sklearn.preprocessing import StandardScaler
        
        # Load scaler
        scaler = joblib.load('models/scaler_primary.pkl')
        
        # Scale input
        input_scaled = scaler.transform([[income, spending]])
        
        # Predict
        cluster = model.predict(input_scaled)[0]
        
        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; border-radius: 15px; text-align: center; color: white;">
                <h1>Cluster {cluster}</h1>
                <p style="font-size: 1.2rem; margin: 10px 0;">Customer Segment</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Show cluster characteristics
            cluster_df = df[df['Cluster'] == cluster]
            st.markdown("### 📊 Segment Characteristics")
            st.write(f"**Avg Age:** {cluster_df['Age'].mean():.1f} years")
            st.write(f"**Avg Income:** ${cluster_df['Annual Income (k$)'].mean():.1f}k")
            st.write(f"**Avg Spending:** {cluster_df['Spending Score (1-100)'].mean():.1f}")
            st.write(f"**Segment Size:** {len(cluster_df)} customers")
        
        # Visualization
        st.markdown("### 📍 Your Position in Segments")
        fig = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)',
                        color='Cluster',
                        title='Your Customer Segment',
                        color_continuous_scale='Viridis')
        
        # Add the new point
        fig.add_trace(go.Scatter(
            x=[income], y=[spending],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star', line=dict(color='white', width=2)),
            name='Your Input',
            showlegend=True
        ))
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🛍️ <strong>Customer Segmentation Dashboard</strong> | K-Means Clustering on Mall Customers</p>
    <p>Built with Streamlit • Python • Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
