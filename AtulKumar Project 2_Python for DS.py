import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("renttherunway.csv")
print(data.head()) 
print(data.describe()) 
user_ids = data['user_id'].unique()
print(f"Unique user IDs: {user_ids}")
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
if len(data.duplicated()) > 0:
  print("Duplicate records found. Dropping them...")
  data.drop_duplicates(inplace=True)
else:
  print("No duplicate records found.")
def clean_weight(weight):
  try:
    return float(weight.strip("lbs"))
  except ValueError:
    return None  
data['weight'] = data['weight'].apply(clean_weight)
data.loc[data['rented for'] == 'party: cocktail', 'rented for'] = 'party'
def convert_height(height):
  try:
    feet, inches = height.strip('"').split("'")
    return float(feet) * 12 + float(inches)
  except ValueError:
    return None  
data['height'] = data['height'].apply(convert_height)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
data.describe(include='all') 
for col in data.select_dtypes(include=['object']):
  print(data[col].value_counts())  
data['rented for'].value_counts().plot(kind='bar')
plt.xlabel('Rented For Category')
plt.ylabel('Count')
plt.title('Distribution of Rented For Categories')
plt.show()
le = LabelEncoder()
for col in data.select_dtypes(include=['object']):
  data[col] = le.fit_transform(data[col])
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
pca = PCA(n_components=0.95)
pca_data = pca.fit_transform(data_scaled)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance.sum())
plt.plot(explained_variance_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()
from sklearn.cluster import KMeans
inertias = []
for k in range(1, 11):
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(pca_data)
  inertias.append(kmeans.inertia_)
plt.plot(range(1, 11), inertias)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Plot for K-Means Clustering')
plt.show()
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(pca_data)
data['kmeans_cluster'] = kmeans.labels_
silhouette_score(pca_data, kmeans.labels_)
from scipy.cluster.hierarchy import dendrogram, linkage
sampled_data = data_scaled.copy()[:1000]  
linkage_matrix = linkage(sampled_data, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Dendrogram for Agglomerative Clustering')
plt.show()
agglo = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
agglo.fit(pca_data)
data['agglo_cluster'] = agglo.labels_
silhouette_score(pca_data, agglo.labels_)
print(pd.crosstab(data['kmeans_cluster'], data['rented for']))