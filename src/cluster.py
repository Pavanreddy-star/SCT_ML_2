import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
# Ensure the src module is found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import load_and_preprocess  # Now it works!

# Load preprocessed data
X, df = load_and_preprocess("data/customer_data.csv")  # Corrected path

# Ensure the results directory exists
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Determine optimal number of clusters using Elbow Method
def elbow_method():
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    # Plot Elbow Graph
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertia, marker='o')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.savefig(f"{results_dir}/elbow_plot.png")  # Save the plot
    plt.show()

# Apply K-Means clustering
def apply_kmeans(n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Save clustered data
    output_file = f"{results_dir}/clustered_customers.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Customer Segmentation Completed! Results saved at {output_file}")

if __name__ == "__main__":
    elbow_method()  # Run elbow method to determine clusters
    apply_kmeans(n_clusters=5)  # Apply K-Means with 5 clusters
