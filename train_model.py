# train_model.py
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import os

def train_and_save():
    print("Загрузка и обработка данных...")

    df = pd.read_csv("adult 3 (1).csv")
    df = df.replace('?', pd.NA).dropna()

    # Кодирование категориальных признаков
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'gender', 'native-country']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Целевая переменная
    le_income = LabelEncoder()
    df['income_encoded'] = le_income.fit_transform(df['income'])

    X = df.drop(['income', 'income_encoded'], axis=1)
    y = df['income_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Стандартизация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA
    pca_2d = PCA(n_components=2)
    pca_3d = PCA(n_components=3)
    X_test_pca2 = pca_2d.fit_transform(X_test_scaled)
    X_test_pca3 = pca_3d.fit_transform(X_test_scaled)

    # Сохраняем всё
    os.makedirs("models", exist_ok=True)

    model_data = {
        'scaler': scaler,
        'pca_2d': pca_2d,
        'pca_3d': pca_3d,
        'X_test_scaled': X_test_scaled,
        'X_test_pca2': X_test_pca2,
        'X_test_pca3': X_test_pca3,
        'X_test': X_test,
        'y_test': y_test,
        'le_income': le_income,
        'feature_names': X.columns.tolist()
    }

    joblib.dump(model_data, "models/decision_tree_model.joblib")
    print("Модель и данные сохранены в models/decision_tree_model.joblib")

if __name__ == "__main__":
    train_and_save()