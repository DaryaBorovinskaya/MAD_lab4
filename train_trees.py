import joblib
from sklearn.tree import DecisionTreeClassifier
import os

data = joblib.load('models/preprocessors.joblib')
X_train = data['X_train'] 
y_train = data['y_train']
feature_names = data['feature_names']

for depth in range(2, 9):
    clf = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=5,
        random_state=42
    )
    clf.fit(X_train, y_train)

    path = f'models/tree_depth_{depth}.joblib'
    joblib.dump(clf, path)

print("Все деревья обучены и сохранены")