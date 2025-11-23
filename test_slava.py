import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')

# --- Загрузка данных ---
file_path = './adult 3 (1).csv'
df = pd.read_csv(file_path)
df = df.replace('?', pd.NA).dropna()

numeric_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
df_analysis = df.copy()
df_analysis['income_num'] = (df_analysis['income'] == '>50K').astype(int)

# --- Категориальные колонки ---
all_cols = df_analysis.columns.tolist()
categorical_cols = [col for col in all_cols if col not in numeric_cols + ['income', 'income_num']]
categorical_cols = [
    col for col in categorical_cols
    if df_analysis[col].dtype == 'object' and 2 <= df_analysis[col].nunique() <= 100
]
fallback = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
if not categorical_cols:
    categorical_cols = [col for col in fallback if col in df_analysis.columns]
if not categorical_cols:
    categorical_cols = ['sex']

# --- Гистограммы числовых признаков ---
rows, cols = 2, 3
fig_hist = make_subplots(
    rows=rows, cols=cols,
    subplot_titles=numeric_cols,
    horizontal_spacing=0.08,
    vertical_spacing=0.15,
)
for i, col in enumerate(numeric_cols):
    data = df_analysis[col].dropna()
    if data.empty:
        continue
    count_intervals = max(1, int(1 + math.log2(len(data))))
    row_pos = (i // cols) + 1
    col_pos = (i % cols) + 1
    fig_hist.add_trace(
        go.Histogram(
            x=data,
            nbinsx=count_intervals,
            marker_color='skyblue',
            marker_line_color='black',
            marker_line_width=1,
            showlegend=False,
        ),
        row=row_pos, col=col_pos
    )
    fig_hist.update_xaxes(title_text="Значение", row=row_pos, col=col_pos)
    fig_hist.update_yaxes(title_text="Частота", row=row_pos, col=col_pos)
fig_hist.update_layout(
    title={'text': "Распределения числовых признаков", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16}},
    showlegend=False,
    height=700,
    width=1400,
)

# --- Матрица корреляции ---
corr_data = df_analysis[numeric_cols]
correlation_matrix = corr_data.corr(method='pearson')
labels = numeric_cols
fig_corr = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=labels,
    y=labels,
    colorscale='RdBu_r',
    text=correlation_matrix.round(2).values,
    texttemplate="%{text}",
    textfont={"size": 12},
    zmin=-1, zmax=1,
    colorbar=dict(title="Корреляция")
))
fig_corr.update_layout(
    title={'text': "Матрица корреляции числовых признаков", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16}},
    xaxis_title="Признаки",
    yaxis_title="Признаки",
    height=600,
    xaxis={'side': 'bottom', 'tickangle': 45},
    yaxis={'side': 'left', 'autorange': 'reversed'},
)

# --- Dash App ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Визуализация данных (Dashboard)", style={'textAlign': 'center'}),

    html.H2("Гистограммы распределений", style={'textAlign': 'center'}),
    dcc.Graph(figure=fig_hist),

    html.H2("Матрица корреляции", style={'textAlign': 'center'}),
    dcc.Graph(figure=fig_corr, style={'height': '600px'}),

    # Однофакторная зависимость
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

    # Двухфакторная зависимость
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
])


# --- Callback: однофакторный ---
@app.callback(
    Output('factor-dependency-graph', 'figure'),
    Input('factor-dropdown', 'value')
)
def update_factor_dependency_plot(selected_factor):
    temp_df = df_analysis.dropna(subset=[selected_factor, 'income_num']).copy()
    if temp_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных", showarrow=False, font_size=20)
        return fig
    grouped = temp_df.groupby(selected_factor).agg(
        mean_income=('income_num', 'mean'),
        count=('income_num', 'count')
    ).reset_index()
    grouped.rename(columns={'mean_income': 'income_num'}, inplace=True)
    grouped = grouped.sort_values('income_num', ascending=False)
    fig = px.bar(
        grouped,
        x=selected_factor,
        y='income_num',
        title=f"Доход (>50K) в зависимости от '{selected_factor}'",
        labels={'income_num': 'Доля >50K'},
        hover_data=['count'],  # ← при наведении видно количество
        color='income_num',
        color_continuous_scale='Blues'
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    return fig


# --- Callback: двухфакторный (столбчатый) ---
@app.callback(
    Output('two-factor-bar', 'figure'),
    Input('factor1-dropdown', 'value'),
    Input('factor2-dropdown', 'value')
)
def update_two_factor_bar(factor1, factor2):
    if factor1 == factor2:
        temp_df = df_analysis.dropna(subset=[factor1, 'income_num']).copy()
        if temp_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="Нет данных", showarrow=False, font_size=20)
            return fig
        grouped = temp_df.groupby(factor1).agg(
            mean_income=('income_num', 'mean'),
            count=('income_num', 'count')
        ).reset_index()
        grouped.rename(columns={'mean_income': 'income_num'}, inplace=True)
        fig = px.bar(
            grouped,
            x=factor1,
            y='income_num',
            title=f"Выбран один и тот же признак",
            labels={'income_num': 'Доля >50K'},
            hover_data=['count'],
            height=500
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig

    temp_df = df_analysis.dropna(subset=[factor1, factor2, 'income_num']).copy()
    if temp_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных", showarrow=False, font_size=20)
        return fig

    grouped = temp_df.groupby([factor1, factor2]).agg(
        mean_income=('income_num', 'mean'),
        count=('income_num', 'count')
    ).reset_index()
    grouped.rename(columns={'mean_income': 'income_num'}, inplace=True)
    grouped = grouped.sort_values(factor1)

    fig = px.bar(
        grouped,
        x=factor1,
        y='income_num',
        color=factor2,
        barmode='group',
        title=f"Доход (>50K) в зависимости от '{factor1}' и '{factor2}'",
        labels={'income_num': 'Доля >50K'},
        hover_data=['count'],  # ← ключевое: видно количество в подсказке
        height=500
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title=factor1,
        yaxis_title="Доля >50K",
        legend_title=factor2
    )
    return fig


if __name__ == '__main__':
    app.run(debug=True)