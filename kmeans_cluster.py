from sklearn.cluster import KMeans
import pandas as pd
import joblib

# 读取一个文件
data = pd.read_csv("key_point.csv")

# 提取特征和目标
features = data.iloc[:, 1:]
target = data.iloc[:, 0]

# 打印数据
print(features.values)
print(target.values)

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(features.values)

# 保存模型
joblib.dump(kmeans, "Model/KMeansModel.joblib")
