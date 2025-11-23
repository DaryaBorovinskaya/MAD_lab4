import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import joblib # pip install joblib

# --- Загрузка и подготовка данных ---
file_path = './adult 3 (1).csv'
df = pd.read_csv(file_path)
df = df.replace('?', pd.NA).dropna()

numeric_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
df_analysis = df.copy()
df_analysis['income_num'] = (df_analysis['income'] == '>50K').astype(int)

# Подготовка данных для кластеризации
df_cluster = df_analysis.drop(columns=['income', 'income_num'])
X_cluster = df_cluster[numeric_cols]

# Стандартизация
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# --- Вычисления для разных n_clusters ---
n_clusters_range = range(2, 6)
results = {}

for n_clusters in n_clusters_range:
    print(f"Обработка для n_clusters={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)

    silhouette_avg = silhouette_score(X_cluster_scaled, cluster_labels)
    sample_silhouette_values = silhouette_samples(X_cluster_scaled, cluster_labels)

    # Сохраняем результаты для данного n_clusters
    results[n_clusters] = {
        'model': kmeans, # Объект обученной модели
        'labels': cluster_labels,
        'silhouette_avg': silhouette_avg,
        'silhouette_samples': sample_silhouette_values,
        'scaler': scaler # Сохраняем scaler, если он понадобится позже
    }
    print(f"  Средний силуэт для {n_clusters} кластеров: {silhouette_avg:.3f}")

# --- Сохранение результатов ---
# Сохраняем словарь с результатами в файл
joblib.dump(results, 'cluster_results.pkl')
joblib.dump(scaler, 'scaler.pkl') 