import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def prepare_and_save_preprocessors():
    df = pd.read_csv('adult 3 (1).csv')
    df = df.replace('?', pd.NA).dropna().drop(columns=['fnlwgt',])

    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'gender', 'native-country']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    le_income = LabelEncoder()
    le_income = LabelEncoder()
    df['income_encoded'] = le_income.fit_transform(df['income'])

    X = df.drop(['income', 'income_encoded'], axis=1)
    y = df['income_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca_2d = PCA(n_components=2).fit(X_test_scaled)
    pca_3d = PCA(n_components=3).fit(X_test_scaled)

    X_test_pca2 = pca_2d.transform(X_test_scaled)
    X_test_pca3 = pca_3d.transform(X_test_scaled)

    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'X_test_scaled': X_test_scaled,
        'pca_2d': pca_2d,
        'pca_3d': pca_3d,
        'X_test_pca2': X_test_pca2,
        'X_test_pca3': X_test_pca3,
        'feature_names': X.columns.tolist(),
        'le_income': le_income
    }, 'models/preprocessors.joblib')

    print("Предобработка завершена")

if __name__ == '__main__':
    prepare_and_save_preprocessors()