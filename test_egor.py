# app.py
import dash
import pandas as pd
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import joblib
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

# === Загрузка данных ===
prep = joblib.load('models/preprocessors.joblib')
X_test = prep['X_test']
y_test = prep['y_test']
X_test_pca2 = prep['X_test_pca2']
X_test_pca3 = prep['X_test_pca3']
feature_names = prep['feature_names']

# Цвета
COLOR_GT50K = '#3498db'   # синий
COLOR_LE50K = '#e74c3c'   # красный

app = dash.Dash(__name__)
app.title = "Adult Income → Дерево решений"

app.layout = html.Div([
    html.H1("Дерево решений + ROC + PCA",
            style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '40px 0'}),

    html.Div([
        html.Label("Глубина дерева:", style={'fontWeight': 'bold', 'fontSize': 18}),
        dcc.Dropdown(id='depth-dropdown',
                     options=[{'label': f'Глубина {d}', 'value': d} for d in range(2, 9)],
                     value=4, clearable=False,
                     style={'width': '400px', 'margin': '20px auto'})
    ], style={'textAlign': 'center'}),

    html.Div(id='dashboard')
], style={'fontFamily': 'Arial', 'backgroundColor': '#f8f9fa', 'padding': '20px'})


# ==================== ЧИСТО PLOTLY ДЕРЕВО ====================
def plotly_tree(model, feature_names, class_names=["≤50K", ">50K"], max_depth=4):
    tree = model.tree_
    fig = go.Figure()

    node_traces = []
    edge_traces = []

    def recurse(node_id, x, y, dx, parent_x=None, parent_y=None):
        value = tree.value[node_id][0]
        total = value.sum()
        if total == 0:
            return

        # Цвет
        majority_class = 1 if value[1] > value[0] else 0
        color = "#e74c3c" if majority_class == 0 else "#3498db"

        # Текст
        if tree.children_left[node_id] == tree.children_right[node_id] == -1:
            text = (
                f"<b>{class_names[majority_class]}</b><br>"
                f"gini = {tree.impurity[node_id]:.3f}"
            )
        else:
            feat = feature_names[tree.feature[node_id]]
            thr = tree.threshold[node_id]
            text = (
                f"{feat} ≤ {thr:.3f}<br>"
                f"gini = {tree.impurity[node_id]:.3f}"
            )

        # === Сначала сохраняем узел в node_traces ===
        node_traces.append(go.Scatter(
            x=[x], y=[y],
            text=[text],
            mode="markers+text",
            textposition="middle center",
            marker=dict(
                size=110,
                color=color,
                line=dict(width=3, color="#333")
            ),
            textfont=dict(size=11, color="#000"),
            hoverinfo="text",
            showlegend=False,
        ))

        # === Линия к родителю → сохраняем отдельно ===
        if parent_x is not None:
            edge_traces.append(go.Scatter(
                x=[parent_x, x], y=[parent_y, y],
                mode="lines",
                line=dict(color="#555", width=2),
                hoverinfo="none",
                showlegend=False,
            ))

        # Рекурсия
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left != -1:
            recurse(left, x - dx, y - 1, dx * 0.65, x, y)
        if right != -1:
            recurse(right, x + dx, y - 1, dx * 0.65, x, y)

    recurse(0, x=0, y=0, dx=1.2)

    # === ВАЖНО: сначала линии, затем узлы ===
    for tr in edge_traces:
        fig.add_trace(tr)
    for tr in node_traces:
        fig.add_trace(tr)

    fig.update_layout(
        height=800,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode="closest",
        autosize=True
    )
    return fig



# ==================== CALLBACK ====================
@callback(
    Output('dashboard', 'children'),
    Input('depth-dropdown', 'value')
)
def update_dashboard(depth):
    # Загружаем предобученное дерево
    model = joblib.load(f'models/tree_depth_{depth}.joblib')

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)

    # 1. Дерево решений — чисто Plotly
    tree_fig = plotly_tree(model, feature_names)

    # 2. Основной дашборд 2×2
    main_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"ROC-кривая (AUC = {roc_auc:.3f})",
            "Матрица ошибок",
        ),
        vertical_spacing=0.15, horizontal_spacing=0.12
    )

    # ROC
    main_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                  line=dict(width=4, color='#e67e22')), row=1, col=1)
    main_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                  line=dict(dash='dash', color='gray')), row=1, col=1)

    # Confusion Matrix
    main_fig.add_trace(go.Heatmap(z=cm, text=cm, texttemplate="%{text}",
                                  colorscale='Blues', showscale=False,
                                  x=['≤50K', '>50K'], y=['≤50K', '>50K']), row=1, col=2)

    # PCA 2D
    pca2_true = px.scatter(
        x=X_test_pca2[:, 0], y=X_test_pca2[:, 1],
        color=y_test.map({0: '<=50K', 1: '>50K'}),
        labels={'x': 'PC1', 'y': 'PC2', 'color': 'Доход (факт)'},
        title='PCA 2D: Истинные метки',
        color_discrete_sequence=[COLOR_LE50K, COLOR_GT50K]
    )

    # Предсказанные метки
    pca2_pred = px.scatter(
        x=X_test_pca2[:, 0], y=X_test_pca2[:, 1],
        color=pd.Series(y_pred).map({0: '<=50K', 1: '>50K'}),
        labels={'x': 'PC1', 'y': 'PC2', 'color': 'Доход (предсказание)'},
        title='PCA 2D: Предсказания дерева',
        color_discrete_sequence=[COLOR_LE50K, COLOR_GT50K]
    )

    # 3D PCA
    fig_3d = px.scatter_3d(x=X_test_pca3[:,0], y=X_test_pca3[:,1], z=X_test_pca3[:,2],
                           color=y_test.map({0:'≤50K', 1:'>50K'}),
                           color_discrete_map={'≤50K': COLOR_LE50K, '>50K': COLOR_GT50K},
                           labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3', 'color': 'Доход'})
    fig_3d.update_traces(marker=dict(size=4))
    fig_3d.update_layout(height=700, scene_camera=dict(eye=dict(x=1.7, y=1.7, z=1.3)))

    # Легенда
    legend = html.Div([
        html.Strong("Цветовая легенда для всех графиков:"),
        html.Div([
            html.Span("●", style={'color': COLOR_LE50K, 'fontSize': 30, 'margin': '0 10px'}),
            html.Span("≤50K", style={'fontSize': 18}),
            html.Span("  "),
            html.Span("●", style={'color': COLOR_GT50K, 'fontSize': 30, 'margin': '0 10px'}),
            html.Span(">50K", style={'fontSize': 18})
        ], style={'margin': '20px', 'textAlign': 'center'})
    ], style={'textAlign': 'center', 'padding': '20px', 'background': 'white',
              'borderRadius': '10px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
              'margin': '40px auto', 'width': '600px'})

    return html.Div([
        legend,

        html.H2("Дерево решений", style={'textAlign': 'center', 'margin': '50px 0 20px'}),
        dcc.Graph(figure=tree_fig),

        html.H2("Метрики и проекции данных", style={'textAlign': 'center', 'margin': '60px 0 30px'}),
        dcc.Graph(figure=main_fig),

        html.Div([
            dcc.Graph(figure=pca2_true, style={"height": "500px", "width": "50%"}),
            dcc.Graph(figure=pca2_pred, style={"height": "500px", "width": "50%"})
        ], style={
            "display": "flex",
            "flexDirection": "row",
            "justifyContent": "space-between"
        }),

        html.H2("3D-визуализация после PCA", style={'textAlign': 'center', 'margin': '70px 0 30px'}),
        dcc.Graph(figure=fig_3d),
    ], style={'maxWidth': '1500px', 'margin': '0 auto'})


if __name__ == '__main__':
    app.run(debug=True)
