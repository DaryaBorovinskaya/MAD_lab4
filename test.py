import dash
import dash_core_components as dcc
import dash_html_components as html
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


file_path = './adult 3 (1).csv'
df = pd.read_csv(file_path)
df = df.replace('?', pd.NA).dropna()
# Create the app
app = dash.Dash(__name__)

numeric_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
df_analysis = df.copy()
df_analysis['income_num'] = (df_analysis['income'] == '>50K').astype(int)



# --- Создание подграфиков (subplots) ---
# Рассчитываем количество строк и столбцов для сетки
rows = 2
cols = 3

fig = make_subplots(
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

    # Добавляем гистограмму
    fig.add_trace(
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

    

    # Настройка подписей осей для каждого подграфика
    fig.update_xaxes(title_text="Значение", row=row_pos, col=col_pos)
    fig.update_yaxes(title_text="Частота", row=row_pos, col=col_pos)

# Общий заголовок
fig.update_layout(
    title={
        'text': "Распределения числовых признаков",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16}
    },
    showlegend=False,
    height=700,  # Увеличиваем высоту, чтобы графики не были слишком маленькими
    width=1400,  # Ширина для 3 столбцов
)


corr_data = df_analysis[numeric_cols]
correlation_matrix = corr_data.corr(method='pearson') # Можно использовать 'spearman' или 'kendall'
labels = numeric_cols  # Порядок признаков

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
    # width=800, # <-- Закомментировано! Пусть Plotly определяет ширину автоматически
    xaxis={'side': 'bottom', 'tickangle': 45},
    yaxis={'side': 'left', 'autorange': 'reversed'},
    margin=dict(l=50, r=50, t=80, b=50), # Опционально: добавляем отступы, чтобы подписи не обрезались
)


# --- Dash layout ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Визуализация данных (Dashboard)", style={'textAlign': 'center'}),

    # Блок с гистограммами
    html.H2("Гистограммы распределений", style={'textAlign': 'center'}),
    dcc.Graph(id='multi-histogram-graph', figure=fig),

    # Блок с тепловой картой
    html.H2("Матрица корреляции", style={'textAlign': 'center'}),
    dcc.Graph(id='correlation-heatmap', figure=fig_corr, style={'height': '600px'}),
])


if __name__ == '__main__':
   app.run(debug=True) # Run the Dash app