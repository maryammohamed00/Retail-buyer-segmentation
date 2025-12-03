import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 4 — VISUALIZATION CODE

# Load the clustered dataset created from Step 2
data = pd.read_csv("clustered_data.csv")

# Make Seaborn plots look cleaner
sns.set(style="whitegrid")

# ==========================================
# 4.1 — Bar Chart: Number of customers per cluster
# ==========================================

cluster_counts = data["cluster"].value_counts()

plt.figure(figsize=(6, 4))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
plt.title("Number of Customers per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count of Customers")
plt.show()

# ==========================================
# 4.2 — Boxplot: Annual Income by Cluster
# ==========================================

plt.figure(figsize=(6, 4))
sns.boxplot(x="cluster", y="annual_income", data=data)
plt.title("Annual Income by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Annual Income")
plt.show()

# ==========================================
# 4.3 — Spending Comparisons by Cluster
# ==========================================

spending_cols = [
    'spend_wine', 'spend_meat', 'spend_fruits',
    'spend_fish', 'spend_sweets', 'spend_gold'
]

# Better visualization: melt into long format
data_melted = data.melt(
    id_vars="cluster",
    value_vars=spending_cols,
    var_name="Category",
    value_name="Spending"
)

plt.figure(figsize=(12, 6))
sns.boxplot(x="Category", y="Spending", hue="cluster", data=data_melted)
plt.title("Spending Distribution per Category by Cluster")
plt.xticks(rotation=45)
plt.ylabel("Amount Spent")
plt.show()

# Individual barplots (optional but useful)
for col in spending_cols:
    plt.figure(figsize=(6, 4))
    sns.barplot(x="cluster", y=col, data=data)
    plt.title(f"{col} by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel(col)
    plt.show()

# ==========================================
# 4.4 — Activity: Days Since Last Purchase by Cluster
# ==========================================

plt.figure(figsize=(6, 4))
sns.boxplot(x="cluster", y="days_since_last_purchase", data=data)
plt.title("Recency (Days Since Last Purchase) by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Days Since Last Purchase")
plt.show()

# ==========================================
# 4.5 — Heatmap: Average Numeric Feature Values per Cluster
# ==========================================

# Select only numeric features to avoid errors
numeric_cols = data.select_dtypes(include='number').columns
cluster_means = data.groupby("cluster")[numeric_cols].mean()

plt.figure(figsize=(14, 8))
sns.heatmap(cluster_means, annot=True, cmap="coolwarm", fmt=".1f")
plt.title("Cluster Profile Heatmap (Numeric Features Only)")
plt.show()

print("Step 4 complete: Visualizations generated successfully.")
