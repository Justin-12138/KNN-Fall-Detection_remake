import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data = pd.read_csv("key_point.csv")
features = data.iloc[:, 1:]
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(features.values)
labels = kmeans.labels_
plt.scatter(features.values[:, 0], features.values[:, 1], c=labels, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.5)
plt.show()
joblib.dump(kmeans, "Model/KMeansModel.joblib")
