import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_curve, auc, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
import os

# --- Загрузка предобученных данных ---
MODEL_PATH = "models/decision_tree_model.joblib"

if not os.path.exists(MODEL_PATH):
    print("Модель не найдена! Запустите: python train_model.py")
    exit()

data = joblib.load(MODEL_PATH)

scaler = data['scaler']
pca_2d = data['pca_2d']
pca_3d = data['pca_3d']
X_test_scaled = data['X_test_scaled']
X_test_pca2 = data['X_test_pca2']
X_test_pca3 = data['X_test_pca3']
X_test = data['X_test']
y_test = data['y_test']
le_income = data['le_income']
feature_names = data['feature_names']

# --- Функции ---
def plot_tree_img(depth):
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=42)
    clf.fit(X_test, y_test)  # обучаем на тесте для визуализации (можно и на train)

    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(clf, feature_names=feature_names, class_names=['<=50K', '>50K'],
              filled=True, rounded=True, fontsize=9, ax=ax)
    plt.title(f"Дерево решений (глубина = {depth})")

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def get_predictions(depth):
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=42)
    clf.fit(X_test, y_test)  # можно и на train, но для простоты
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)

    return y_pred, y_pred_proba, fpr, tpr, roc_auc, cm

# --- Dash ---
app = dash.Dash(__name__)
app.title = "Adult Income — Интерактивный анализ"

app.layout = html.Div([
    html.H1("Анализ дохода с деревом решений", style={'textAlign': 'center', 'margin': '30px'}),

    html.Div([
        html.Label("Выберите глубину дерева:", style={'fontWeight': 'bold', 'fontSize': 18}),
        dcc.Dropdown(
            id='depth-dropdown',
            options=[{'label': f'Глубина {i}', 'value': i} for i in [2, 3, 4, 5, 6, 7, 8]],
            value=4,
            clearable=False,
            style={'width': '50%', 'margin': '20px auto'}
        ),
    ], style={'textAlign': 'center'}),

    html.Hr(),

    html.Div(id='tree-image', style={'textAlign': 'center', 'margin': '30px'}),

    html.Div([
        html.Div([html.H3("ROC-кривая"), dcc.Graph(id='roc-graph')], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([html.H3("Матрица ошибок"), dcc.Graph(id='cm-graph')], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
    ]),

    html.Div([
        html.H3("PCA визуализация", style={'textAlign': 'center', 'marginTop': '40px'}),
        html.Div([
            dcc.Graph(id='pca-2d-true', style={'width': '48%'}),
            dcc.Graph(id='pca-2d-pred', style={'width': '48%'}),
        ], style={'display': 'flex', 'justifyContent': 'space-around'}),
        dcc.Graph(id='pca-3d', style={'height': '600px', 'marginTop': '20px'})
    ])
])

# --- Callback ---
@app.callback(
    [Output('tree-image', 'children'),
     Output('roc-graph', 'figure'),
     Output('cm-graph', 'figure'),
     Output('pca-2d-true', 'figure'),
     Output('pca-2d-pred', 'figure'),
     Output('pca-3d', 'figure')],
    Input('depth-dropdown', 'value')
)
def update_all(depth):
    tree_b64 = plot_tree_img(depth)
    tree_img = html.Img(src=f'data:image/png;base64,{tree_b64}', style={'maxWidth': '100%', 'border': '2px solid #ddd'})

    y_pred, y_pred_proba, fpr, tpr, roc_auc, cm = get_predictions(depth)

    # ROC
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'AUC = {roc_auc:.3f}', line=dict(width=3)))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='gray'), showlegend=False))
    roc_fig.update_layout(title=f'ROC-кривая (глубина = {depth})', template='plotly_white')

    # Confusion Matrix
    cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                       x=['<=50K', '>50K'], y=['<=50K', '>50K'])
    cm_fig.update_layout(title=f'Матрица ошибок (глубина = {depth})')

    # PCA
    pca2_true = px.scatter(x=X_test_pca2[:,0], y=X_test_pca2[:,1],
                           color=y_test.map({0: '<=50K', 1: '>50K'}),
                           title='PCA 2D: Реальные метки')
    pca2_pred = px.scatter(x=X_test_pca2[:,0], y=X_test_pca2[:,1],
                           color=pd.Series(y_pred).map({0: '<=50K', 1: '>50K'}),
                           title='PCA 2D: Предсказания дерева')
    pca3 = px.scatter_3d(x=X_test_pca3[:,0], y=X_test_pca3[:,1], z=X_test_pca3[:,2],
                         color=y_test.map({0: '<=50K', 1: '>50K'}),
                         title='PCA 3D')

    return tree_img, roc_fig, cm_fig, pca2_true, pca2_pred, pca3

if __name__ == '__main__':
    print("Запуск дашборда: http://127.0.0.1:8050")
    app.run(debug=False)
