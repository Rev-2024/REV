import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Standardize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply K-Means Clustering (2 clusters: malignant and benign)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the DataFrame
df['Cluster'] = clusters

# Reduce dimensions for visualization using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Plot clustering results
plt.figure(figsize=(6, 5))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette='Set1', s=60)
plt.title("K-Means Clustering on Breast Cancer Data (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Compare with actual target labels
from sklearn.metrics import accuracy_score
labels = np.where(clusters == 1, 0, 1)  # Flip labels if needed
accuracy = accuracy_score(data.target, labels)
print(f"Clustering Accuracy (approx.): {accuracy:.2f}")
