import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
import numpy as np
import math
import dash
from dash import dcc, html, Input, Output, callback
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


print("=== Запуск скрипта дашборда ===")

file_path = './adult 3 (1).csv'
df = pd.read_csv(file_path)
df = df.replace('?', pd.NA).dropna()


numeric_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
df_analysis = df.copy()
df_analysis['income_num'] = (df_analysis['income'] == '>50K').astype(int)

# --- Создание подграфиков (subplots) для гистограмм ---
rows = 2
cols = 3

fig = make_subplots( # Переименовал из fig_hist, т.к. в layout используется fig
    rows=rows, cols=cols,
    subplot_titles=numeric_cols,
    horizontal_spacing=0.08,
    vertical_spacing=0.15,
)

for i, col in enumerate(numeric_cols):
    data = df_analysis[col].dropna()
    n = len(data)

    if n == 0:
        continue

    # Количество интервалов по Стёрджессу
    count_intervals = int(1 + math.log2(df_analysis.shape[0]))
    if count_intervals <= 0:
        count_intervals = 1

    row_pos = (i // cols) + 1
    col_pos = (i % cols) + 1

    fig.add_trace( # Переименовал из fig_hist.add_trace
        go.Histogram(
            x=data,
            nbinsx=count_intervals,
            marker_color='skyblue',
            marker_line_color='black',
            marker_line_width=1,
            name=col,
            showlegend=False,
        ),
        row=row_pos, col=col_pos
    )

    fig.update_xaxes(title_text="Значение", row=row_pos, col=col_pos)
    fig.update_yaxes(title_text="Частота", row=row_pos, col=col_pos)

fig.update_layout( # Переименовал из fig_hist.update_layout
    title={
        'text': "Распределения числовых признаков",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16}
    },
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
    hoverongaps=False,
    zmin=-1,
    zmax=1,
    colorbar=dict(
        title=dict(
            text="Корреляция",
            side="right",
        ),
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=["-1", "-0.5", "0", "0.5", "1"]
    )
))

fig_corr.update_layout(
    title={
        'text': "Матрица корреляции числовых признаков",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16}
    },
    xaxis_title="Признаки",
    yaxis_title="Признаки",
    height=600,
    xaxis={'side': 'bottom', 'tickangle': 45},
    yaxis={'side': 'left', 'autorange': 'reversed'},
    margin=dict(l=50, r=50, t=80, b=50),
)

# --- Матрица диаграмм рассеивания (Scatter Matrix) ---
df_analysis_for_scatter = df_analysis.copy()
df_analysis_for_scatter['income_num'] = df_analysis_for_scatter['income_num'].astype(str)

# Теперь используем преобразованный датафрейм
fig_scatter_px = px.scatter_matrix(
    df_analysis_for_scatter,  # <-- Используем копию с изменённым типом
    dimensions=numeric_cols,
    color='income_num',
    color_discrete_map={'0': 'lightcoral', '1': 'lightgreen'}, # <-- Важно: ключи теперь строки!
    title="Матрица диаграмм рассеивания числовых признаков"
)

fig_scatter_px.update_layout(
    title={
        'text': "Матрица диаграмм рассеивания числовых признаков",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16}
    },
    height=1000,
    width=1000,
    legend=dict(
        title="Доход >50K", # Название легенды
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02
    )
)

# --- Dash layout ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Визуализация данных (Dashboard)", style={'textAlign': 'center'}),

    html.H2("Гистограммы распределений", style={'textAlign': 'center'}),
    dcc.Graph(id='multi-histogram-graph', figure=fig),

    html.H2("Матрица корреляции", style={'textAlign': 'center'}),
    dcc.Graph(id='correlation-heatmap', figure=fig_corr, style={'height': '600px'}),

    html.H2("Матрица диаграмм рассеивания", style={'textAlign': 'center'}), # <-- Изменён заголовок
    dcc.Graph(
        id='scatter-matrix',
        figure=fig_scatter_px,
        style={
            'display': 'block',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'width': '1000px'  # <-- Фиксируем ширину
        }
    ),
])

if __name__ == '__main__':
   app.run(debug=True)