import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import joblib 


file_path = './adult 3 (1).csv'
df = pd.read_csv(file_path)
df = df.replace('?', pd.NA).dropna()

numeric_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
df_analysis = df.copy()
df_analysis['income_num'] = (df_analysis['income'] == '>50K').astype(int)

df_cluster = df_analysis.drop(columns=['income', 'income_num'])
X_cluster = df_cluster[numeric_cols]

scaler = RobustScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

n_clusters_range = range(2, 6)
results = {}

for n_clusters in n_clusters_range:
    print(f"Обработка для n_clusters={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)

    silhouette_avg = silhouette_score(X_cluster_scaled, cluster_labels)
    sample_silhouette_values = silhouette_samples(X_cluster_scaled, cluster_labels)

    results[n_clusters] = {
        'model': kmeans,
        'labels': cluster_labels,
        'silhouette_avg': silhouette_avg,
        'silhouette_samples': sample_silhouette_values,
        'scaler': scaler 
    }
    print(f"  Средний силуэт для {n_clusters} кластеров: {silhouette_avg:.3f}")

joblib.dump(results, 'cluster_results.pkl')
joblib.dump(scaler, 'scaler.pkl') 