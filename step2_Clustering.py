import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


data = pd.read_csv("cleaned_data.csv")

features = [
    'annual_income',
    'total_spend',
    'total_purchases',
    'family_size',
    'age',
    'num_discount_purchases',
    'days_since_last_purchase',
    'web_visits_last_month'
]

X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

##Elbow 
# List to store inertia values for each k
inertia = []

# We test k = 2 to 10 clusters
# Why not start at k=1? -> With 1 cluster, all data is in one group (not useful)
# Why stop at k=10? -> Enough to find the elbow without making the graph too large
K_range = range(2, 11)

# Loop through each possible number of clusters
for k in K_range:

    # Create the KMeans model with k clusters
    # random_state=42 ensures we get the same result every time (reproducible)
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Train K-Means on the scaled data
    kmeans.fit(X_scaled)

    # Save the inertia (how far points are from cluster centers)
    # Lower inertia = tighter clusters
    inertia.append(kmeans.inertia_)

# Plot the elbow graph
plt.plot(K_range, inertia, 'o-')
plt.xlabel("Number of Clusters (k)")       # x-axis: number of clusters
plt.ylabel("Inertia")                      # y-axis: total distance to centers
plt.title("Elbow Method")                  # title
plt.show()

#silhouette 

# We test values of k from 2 to 10
# (k=1 is useless because you cannot measure silhouette with only 1 cluster)
for k in range(2, 11):

    # Create a KMeans model using k clusters
    # random_state=42 ensures we get the same results each time
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Fit KMeans to the scaled data AND return the cluster labels for each sample
    # labels = [0, 1, 0, 2, 1, ...] depending on which cluster each row belongs to
    labels = kmeans.fit_predict(X_scaled)

    # Calculate the silhouette score
    # Silhouette score tells how well-separated and well-formed the clusters are
    # Score range: -1 to +1
    # Higher score = better clusters (more distinct & separated)
    score = silhouette_score(X_scaled, labels)

    # Print the result for this value of k
    print(f"k = {k}, silhouette score = {score}")
    

#Train final K-model 

# Best number of clusters from silhouette and elbow
k = 2  # you can change to 3 or 4 later if you want more segments

# Create final KMeans model
kmeans = KMeans(n_clusters=k, random_state=42)

# Fit on the scaled features and assign a cluster to each customer
data["cluster"] = kmeans.fit_predict(X_scaled)

# See how many customers in each cluster
print("Number of customers in each cluster:")
print(data["cluster"].value_counts())

# STEP 2.7 — Analyze the clusters

print("Cluster Summary (average values for each segment):")
cluster_summary = data.groupby("cluster")[features].mean()
print(cluster_summary)

# STEP 2.8 — Save final dataset


# -----------------------------------
# BONUS: PCA for 2D visualization
# -----------------------------------

# Reduce the 13 features to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Add PC1 and PC2 to the dataframe (optional but useful)
data["PC1"] = X_pca[:, 0]
data["PC2"] = X_pca[:, 1]

# Show how much variance each principal component explains
print("Explained variance ratio by PCA components:", pca.explained_variance_ratio_)
print("Total variance explained by first 2 components:", pca.explained_variance_ratio_.sum())


# Scatter plot of the two principal components, colored by cluster
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    data["PC1"], 
    data["PC2"], 
    c=data["cluster"],    # color by cluster label (0 or 1)
    cmap="viridis",
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segments Visualized in PCA Space")

# Add legend for clusters
legend1 = plt.legend(*scatter.legend_elements(), title="Cluster")
plt.gca().add_artist(legend1)

plt.show()

data.to_csv("clustered_data.csv", index=False)
print("Step 2 complete! File saved as clustered_data.csv")



