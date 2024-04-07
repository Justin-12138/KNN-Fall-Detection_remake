from sklearn.cluster import KMeans
import pandas as pd
import joblib


data = pd.read_csv("key_point.csv")
features = data.iloc[:, 1:]
print(features.values)
print(target.values)
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(features.values)


joblib.dump(kmeans, "Model/KMeansModel.joblib")
