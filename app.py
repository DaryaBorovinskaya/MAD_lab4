from dash import Dash, dcc, html
import joblib
from data_processing import get_processed_data
from visualizations import create_histograms, create_correlation_heatmap, create_scatter_matrix, create_confusion_matrix, create_roc_curve, create_feature_coefficients, create_silhouette_plots
from modeling import train_logistic_regression
from callbacks import register_callbacks


df_analysis, X, y, numeric_cols, categorical_cols = get_processed_data()

fig_histograms = create_histograms(df_analysis, numeric_cols)
fig_corr = create_correlation_heatmap(df_analysis, numeric_cols)
fig_scatter_px = create_scatter_matrix(df_analysis, numeric_cols)

log_reg_results = train_logistic_regression(X, y)
fig_cm = create_confusion_matrix(log_reg_results['cm'], title='Матрица ошибок (Logistic Regression)')
fig_roc = create_roc_curve(log_reg_results['fpr'], log_reg_results['tpr'], log_reg_results['roc_auc'])
fig_coef = create_feature_coefficients(log_reg_results['coefficients'], numeric_cols)

try:
    cluster_results = joblib.load('cluster_results.pkl')
    n_clusters_range = range(2, 6)
    silhouette_figures = create_silhouette_plots(cluster_results, n_clusters_range)
except FileNotFoundError:
    print("Файл 'cluster_results.pkl' не найден. Пожалуйста, сначала запустите 'calculate_clusters.py'.")
    silhouette_figures = {}
    n_clusters_range = range(2, 6)

prep = joblib.load('models/preprocessors.joblib')
feature_names = prep['feature_names'] 

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Визуализация данных (Dashboard)", style={'textAlign': 'center'}),

    html.H2("Гистограммы распределений", style={'textAlign': 'center'}),
    dcc.Graph(id='multi-histogram-graph', figure=fig_histograms),

    html.H2("Матрица корреляций", style={'textAlign': 'center'}),
    dcc.Graph(id='correlation-heatmap', figure=fig_corr, style={'height': '600px'}),

    html.H2("Матрица диаграмм рассеивания", style={'textAlign': 'center'}),
    dcc.Graph(
        id='scatter-matrix',
        figure=fig_scatter_px,
        style={
            'display': 'block',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'width': '1000px'
        }
    ),

    html.H2("Модель: Логистическая регрессия", style={'textAlign': 'center'}),

    html.H3("Уравнение модели:", style={'textAlign': 'center'}),
    html.Div([
        html.Pre(log_reg_results['equation'], style={
            'textAlign': 'left',
            'margin': '20px auto',
            'padding': '10px',
            'backgroundColor': '#f4f4f4',
            'border': '1px solid #ccc',
            'borderRadius': '5px',
            'maxWidth': '1000px',
            'overflowX': 'auto',
            'fontFamily': 'monospace',
            'fontSize': '14px'
        })
    ], style={'textAlign': 'center'}),

    html.H2("Матрица ошибок", style={'textAlign': 'center'}),
    dcc.Graph(
        id='confusion-matrix',
        figure=fig_cm,
        style={
            'display': 'block',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'width': '500px'
        }
    ),

    html.H2("ROC-кривая", style={'textAlign': 'center'}),
    dcc.Graph(
        id='roc-curve',
        figure=fig_roc,
        style={
            'display': 'block',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'width': '500px'
        }
    ),

    html.H2("Коэффициенты признаков", style={'textAlign': 'center'}),
    dcc.Graph(
        id='feature-coefficients',
        figure=fig_coef,
        style={
            'display': 'block',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'width': '800px'
        }
    ),

    html.H2("Зависимость дохода от одного категориального признака", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Признак:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='factor-dropdown',
            options=[{'label': col, 'value': col} for col in categorical_cols],
            value=categorical_cols[0],
            clearable=False,
            style={'width': '300px'}
        )
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '20px'}),
    dcc.Graph(id='factor-dependency-graph'),

    html.H2("Зависимость дохода от двух категориальных признаков", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Фактор 1 (ось X):", style={'fontWeight': 'bold', 'marginRight': '5px'}),
        dcc.Dropdown(
            id='factor1-dropdown',
            options=[{'label': col, 'value': col} for col in categorical_cols],
            value=categorical_cols[0],
            clearable=False,
            style={'width': '200px', 'marginRight': '20px'}
        ),
        html.Label("Фактор 2 (цвет):", style={'fontWeight': 'bold', 'marginRight': '5px'}),
        dcc.Dropdown(
            id='factor2-dropdown',
            options=[{'label': col, 'value': col} for col in categorical_cols],
            value=categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[0],
            clearable=False,
            style={'width': '200px'}
        )
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '20px'}),
    dcc.Graph(id='two-factor-bar'),

    html.H2("Дерево решений", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Глубина дерева:", style={'fontWeight': 'bold', 'fontSize': 18}),
        dcc.Dropdown(id='depth-dropdown',
                     options=[{'label': f'Глубина {d}', 'value': d} for d in range(2, 9)],
                     value=4, clearable=False,
                     style={'width': '400px', 'margin': '20px auto'})
    ], style={'textAlign': 'center'}),

    html.Div(id='dashboard'),

    html.H2("Кластеризация", style={'textAlign': 'center'}),
    html.H2("Анализ силуэта", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Число кластеров:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='n-clusters-dropdown',
            options=[{'label': str(n), 'value': n} for n in n_clusters_range],
            value=n_clusters_range[0],
            searchable=False,
            clearable=False,
            style={'width': '100px'}
        )
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '20px'}),

    html.Div([
        dcc.Graph(
            id='silhouette-plot',
            style={
                'display': 'block',
                'margin-left': 'auto',
                'margin-right': 'auto',
                'width': '700px',
            }
        )
    ], style={'textAlign': 'center'}),
])

register_callbacks(app, df_analysis, silhouette_figures, prep, feature_names)

if __name__ == '__main__':
    app.run(debug=True)