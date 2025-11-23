import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# --- Загрузка данных ---
file_path = './adult 3 (1).csv'
df = pd.read_csv(file_path)
df = df.replace('?', pd.NA).dropna()

numeric_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
df_analysis = df.copy()
df_analysis['income_num'] = (df_analysis['income'] == '>50K').astype(int)

# Encode categorical variables
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                    'relationship', 'race', 'gender', 'native-country']

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# ВАЖНО: кодируем income ДО train_test_split!
le_income = LabelEncoder()
df['income_encoded'] = le_income.fit_transform(df['income'])  # 0 = <=50K, 1 = >50K

X = df.drop(['income', 'income_encoded'], axis=1)
y = df['income_encoded']  # используем уже закодированную переменную!

# Теперь разбиваем
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Стандартизация (обязательно для PCA!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------- Функции для графиков ---------------------
def plot_tree_img(depth):
    clf = DecisionTreeClassifier(
        max_depth=4,  # важно ограничить глубину, иначе дерево огромное
        min_samples_leaf=5,
        random_state=42
    )
    clf.fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(20, 12))  # большой размер
    plot_tree(
        clf,
        feature_names=X.columns.tolist(),
        class_names=['<=50K', '>50K'],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )
    plt.title("Decision Tree (max_depth=4)", fontsize=16)

    # Сохраняем в BytesIO → base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def get_predictions_and_metrics(depth):
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    return clf, y_pred, y_pred_proba, fpr, tpr, roc_auc, cm


# PCA на тестовых данных (для визуализации)
pca_2d = PCA(n_components=2)
pca_3d = PCA(n_components=3)
X_test_pca2 = pca_2d.fit_transform(X_test_scaled)
X_test_pca3 = pca_3d.fit_transform(X_test_scaled)

# --------------------- Dash App ---------------------
app = dash.Dash(__name__)
app.title = "Adult Income — Decision Tree + ROC + Confusion Matrix + PCA"

app.layout = html.Div([
    html.H1("Визуализация дерева решений",
            style={'textAlign': 'center', 'margin': '30px', 'color': '#2c3e50'}),

    html.Div([
        html.Label("Глубина дерева решений:", style={'fontWeight': 'bold'}),
        dcc.Slider(id='depth-slider', min=2, max=8, step=1, value=4,
                   marks={i: str(i) for i in range(2, 9)}, tooltip={"placement": "bottom"})
    ], style={'width': '90%', 'margin': '30px auto'}),

    html.Hr(),

    html.Div([
        # Дерево
        html.Div([
            html.H3("Дерево решений", style={'textAlign': 'center'}),
            html.Div(id='tree-image')
        ], style={'width': '100%', 'marginBottom': '40px'}),

        # ROC + Confusion Matrix бок о бок
        html.Div([
            html.Div([
                html.H3("ROC-кривая", style={'textAlign': 'center'}),
                dcc.Graph(id='roc-graph')
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                html.H3("Матрица ошибок", style={'textAlign': 'center'}),
                dcc.Graph(id='cm-graph')
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
        ], style={'textAlign': 'center'}),

        # PCA 2D и 3D
        html.Div([
            html.H3("4. PCA — визуализация данных в 2D и 3D", style={'textAlign': 'center', 'color': '#34495e'}),

            html.Div([
                dcc.Graph(id='pca-2d-true'),
                dcc.Graph(id='pca-2d-pred')
            ], style={'display': 'flex', 'justifyContent': 'space-around'}),

            html.Div([
                dcc.Graph(id='pca-3d', style={'height': '600px'})
            ], style={'marginTop': '30px', 'textAlign': 'center'})
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)', 'margin': '20px'})
    ], style={'padding': '20px'})
])


# --------------------- Callback ---------------------
@app.callback(
    [Output('tree-image', 'children'),
     Output('roc-graph', 'figure'),
     Output('cm-graph', 'figure'),
     Output('pca-2d-true', 'figure'),
     Output('pca-2d-pred', 'figure'),
     Output('pca-3d', 'figure')],
    Input('depth-slider', 'value')
)
def update_all(depth):
    # 1. Дерево
    tree_b64 = plot_tree_img(depth)
    tree_img = html.Img(src=f'data:image/png;base64,{tree_b64}',
                        style={'maxWidth': '100%', 'height': 'auto', 'border': '1px solid #ccc'})

    # 2. ROC и CM
    clf, y_pred, y_pred_proba, fpr, tpr, roc_auc, cm = get_predictions_and_metrics(depth)

    # ROC Plotly
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 name=f'ROC (AUC = {roc_auc:.3f})',
                                 line=dict(color='darkorange', width=3)))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 name='Random', line=dict(dash='dash', color='navy')))
    roc_fig.update_layout(title=f'ROC-кривая (глубина = {depth})',
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate',
                          template='plotly_white',
                          width=500, height=450)

    # Confusion Matrix Plotly
    cm_fig = px.imshow(cm, text_auto=True, aspect="auto",
                       labels=dict(x="Предсказано", y="Фактически", color="Количество"),
                       x=['<=50K', '>50K'], y=['<=50K', '>50K'],
                       color_continuous_scale='Blues')
    cm_fig.update_layout(title=f'Матрица ошибок (глубина = {depth})',
                         width=500, height=450)

    # 4. PCA визуализации
    # Истинные метки
    pca2_true = px.scatter(
        x=X_test_pca2[:, 0], y=X_test_pca2[:, 1],
        color=y_test.map({0: '<=50K', 1: '>50K'}),
        labels={'x': 'PC1', 'y': 'PC2', 'color': 'Доход (факт)'},
        title='PCA 2D: Истинные метки',
        color_discrete_sequence=['#636efa', '#ef553b']
    )
    pca2_true.update_layout(template='plotly_white')

    # Предсказанные метки
    pca2_pred = px.scatter(
        x=X_test_pca2[:, 0], y=X_test_pca2[:, 1],
        color=pd.Series(y_pred).map({0: '<=50K', 1: '>50K'}),
        labels={'x': 'PC1', 'y': 'PC2', 'color': 'Доход (предсказание)'},
        title='PCA 2D: Предсказания дерева',
        color_discrete_sequence=['#636efa', '#ef553b']
    )
    pca2_pred.update_layout(template='plotly_white')

    # 3D PCA
    pca3_fig = px.scatter_3d(
        x=X_test_pca3[:, 0], y=X_test_pca3[:, 1], z=X_test_pca3[:, 2],
        color=y_test.map({0: '<=50K', 1: '>50K'}),
        labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3', 'color': 'Доход'},
        title='PCA 3D — вращайте мышкой!',
        color_discrete_sequence=['#636efa', '#ef553b']
    )
    pca3_fig.update_layout(scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))

    return tree_img, roc_fig, cm_fig, pca2_true, pca2_pred, pca3_fig


# --------------------- Запуск ---------------------
if __name__ == '__main__':
    app.run(debug=True)