from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np


def train_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
 
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    coefficients = log_reg.coef_[0]
    intercept = log_reg.intercept_[0]

    feature_names = X.columns.tolist() 
    z_equation_str = f"z = {intercept:.4f}"
    for feat, coef in zip(feature_names, coefficients):
        sign = "+" if coef >= 0 else "-"
        z_equation_str += f" {sign} ({abs(coef):.4f} * {feat})"
    full_equation_str = f"P(income >50K | X) = 1 / (1 + exp(-({z_equation_str})))"

    return {
        'model': log_reg,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'cm': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'coefficients': coefficients,
        'equation': full_equation_str
    }
