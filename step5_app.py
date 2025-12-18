import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# LOAD DATA AND MODELS
# ===============================
@st.cache_data
def load_data():
    data = pd.read_csv("clustered_data.csv")
    return data

data = load_data()

# Features used for models
features = [
    'annual_income', 'total_spend', 'total_purchases', 'family_size',
    'age', 'num_discount_purchases', 'days_since_last_purchase', 'web_visits_last_month'
]

X = data[features]
y = data['cluster']

# Scaling for models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train final models (for demo purposes, retrain on all data)
kmeans = KMeans(n_clusters=len(data['cluster'].unique()), random_state=42)
data['cluster'] = kmeans.fit_predict(X_scaled)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

# ===============================
# STREAMLIT UI
# ===============================
st.title("Retail Customer Segmentation Dashboard")
st.write("Predict the customer segment and explore cluster profiles.")

# ------------------------------
# User Input
# ------------------------------
st.sidebar.header("Enter New Customer Info")
def user_input_features():
    input_data = {
        'annual_income': st.sidebar.number_input('Annual Income', min_value=0, value=50000),
        'total_spend': st.sidebar.number_input('Total Spend', min_value=0, value=1000),
        'total_purchases': st.sidebar.number_input('Total Purchases', min_value=0, value=10),
        'family_size': st.sidebar.number_input('Family Size', min_value=0, value=2),
        'age': st.sidebar.number_input('Age', min_value=0, value=35),
        'num_discount_purchases': st.sidebar.number_input('Discount Purchases', min_value=0, value=2),
        'days_since_last_purchase': st.sidebar.number_input('Days Since Last Purchase', min_value=0, value=30),
        'web_visits_last_month': st.sidebar.number_input('Web Visits Last Month', min_value=0, value=5)
    }
    features_df = pd.DataFrame([input_data])
    return features_df

user_df = user_input_features()

# ------------------------------
# Prediction
# ------------------------------
st.subheader("Predicted Cluster for New Customer")
user_cluster = rf.predict(user_df)[0]
st.write(f"Cluster: {user_cluster}")

# ------------------------------
# Show Cluster Profile
# ------------------------------
st.subheader("Cluster Profiles")
cluster_summary = data.groupby("cluster")[features].mean()
st.dataframe(cluster_summary)

# ------------------------------
# Visualizations
# ------------------------------
st.subheader("Cluster Visualizations")

# 1. Customer count per cluster
st.write("Number of Customers per Cluster")
fig, ax = plt.subplots()
sns.countplot(x='cluster', data=data, ax=ax)
st.pyplot(fig)

# 2. Boxplot: Annual Income by Cluster
st.write("Annual Income by Cluster")
fig2, ax2 = plt.subplots()
sns.boxplot(x='cluster', y='annual_income', data=data, ax=ax2)
st.pyplot(fig2)

# 3. Spending by Cluster (heatmap)
st.write("Average Numeric Features per Cluster")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(cluster_summary, annot=True, cmap='coolwarm', fmt=".1f", ax=ax3)
st.pyplot(fig3)
