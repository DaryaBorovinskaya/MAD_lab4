import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Загрузка и предобработка данных ---
file_path = './adult 3 (1).csv'
df = pd.read_csv(file_path)
df = df.replace('?', pd.NA).dropna()

numeric_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_cols = ['workclass', 'education', 'marital-status',
                    'occupation', 'relationship', 'race', 'gender',
                    'native-country']

df_analysis = df.copy()
df_analysis['income_num'] = (df_analysis['income'] == '>50K').astype(int)

X = df_analysis[numeric_cols]
y = df_analysis['income_num']

# --- Инициализация Dash ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# --- Макет дашборда ---
app.layout = dbc.Container(
    [
        html.H1("Классификационный датасет: Adult Income", className="text-center my-4"),

        dbc.Row([
            # Боковая панель с фильтрами
            dbc.Col([
                html.H5("Настройки визуализаций", className="mt-3"),
                html.Label("Гистограмма:"),
                dcc.Dropdown(
                    id='hist-feature',
                    options=[{'label': col, 'value': col} for col in numeric_cols],
                    value=numeric_cols[0],
                    clearable=False,
                    style={'width': '100%'}
                ),
                html.Br(),

                html.Label("Диаграмма рассеивания X:"),
                # Для scatter — теперь выбираем из категориальных признаков
                dcc.Dropdown(
                    id='scatter-x',
                    options=[{'label': col, 'value': col} for col in categorical_cols],
                    value=categorical_cols[0],
                    clearable=False,
                    style={'width': '100%'}
                ),
                html.Label("Y:"),
                dcc.Dropdown(
                    id='scatter-y',
                    options=[{'label': col, 'value': col} for col in categorical_cols],
                    value=categorical_cols[1],
                    clearable=False,
                    style={'width': '100%'}
                ),
                html.Br(),

                html.Label("Boxplot:"),
                dcc.Dropdown(
                    id='box-feature',
                    options=[{'label': col, 'value': col} for col in numeric_cols],
                    value=numeric_cols[0],
                    clearable=False,
                    style={'width': '100%'}
                ),
                html.Br(),

                html.Label("Столбчатая диаграмма:"),
                dcc.Dropdown(
                    id='bar-feature',
                    options=[{'label': col, 'value': col} for col in categorical_cols],
                    value=categorical_cols[0],
                    clearable=False,
                    style={'width': '100%'}
                ),
            ], width=2, style={'background-color': '#f8f9fa', 'padding': '20px', 'border-right': '1px solid #dee2e6'}),

            # Основные графики
            dbc.Col([
                dbc.Row([
                    dbc.Col(dcc.Graph(id='histogram-chart'), width=4),
                    dbc.Col(dcc.Graph(id='pie-chart'), width=4),
                    dbc.Col(dcc.Graph(id='scatter-chart'), width=4),
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='boxplot-chart'), width=6),
                    dbc.Col(dcc.Graph(id='bar-chart'), width=6),
                ], className="mb-4"),
            ], width=10)
        ])
    ],
    fluid=True,
    style={'padding': '20px'}
)

# --- Callbacks ---

@app.callback(
    Output('histogram-chart', 'figure'),
    Input('hist-feature', 'value')
)
def update_histogram(selected_feature):
    fig = px.histogram(
        df_analysis,
        x=selected_feature,
        nbins=50,
        title=f"Гистограмма: {selected_feature}",
        labels={selected_feature: selected_feature}
    )
    
    # Добавляем обводку к столбцам
    fig.update_traces(
        marker=dict(
            line=dict(
                color='black',   # цвет обводки
                width=1          # толщина обводки
            )
        )
    )
    
    fig.update_layout(showlegend=False)
    return fig

@app.callback(
    Output('pie-chart', 'figure'),
    Input('hist-feature', 'value')  # dummy input — чтобы триггерить обновление
)
def update_pie(_):
    income_counts = df_analysis['income'].value_counts().reset_index()
    income_counts.columns = ['income', 'count']
    fig = px.pie(income_counts, names='income', values='count',
                 title="Распределение целевой переменной (income)",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    return fig

@app.callback(
    Output('scatter-chart', 'figure'),
    [Input('scatter-x', 'value'),
     Input('scatter-y', 'value')]
)
def update_scatter_bubble(x_col, y_col):
    # Агрегируем данные: количество и доля >50K по комбинациям
    agg = df_analysis.groupby([x_col, y_col]).agg(
        count=('income_num', 'count'),
        mean_income=('income_num', 'mean')
    ).reset_index()

    # Если среднее == 0 или 1 — немного смещаем, чтобы не было нулевого размера
    agg['size'] = agg['count'].apply(lambda x: max(x, 1))
    agg['color'] = agg['mean_income']

    # Строим bubble chart
    fig = px.scatter(
        agg,
        x=x_col,
        y=y_col,
        size='size',
        color='color',
        hover_data=['count', 'mean_income'],
        title=f"Диаграмма рассеивания: {x_col} vs {y_col}<br><sup>Размер = количество наблюдений, Цвет = доля >50K</sup>",
        labels={x_col: x_col, y_col: y_col},
        color_continuous_scale=px.colors.sequential.Reds,
        size_max=60
    )

    # Добавляем текстовые метки (число наблюдений) внутрь пузырьков
    fig.update_traces(
        text=agg['count'].astype(str),
        textposition='middle center',
        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
        textfont=dict(size=10, color='white')
    )

    # Настройка макета
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        coloraxis_colorbar=dict(title="Доля >50K"),
        showlegend=False,
        height=600
    )

    return fig

@app.callback(
    Output('boxplot-chart', 'figure'),
    Input('box-feature', 'value')
)
def update_boxplot(selected_feature):
    fig = px.box(df_analysis, x='income', y=selected_feature, color='income',
                 title=f"Ящики с усами: {selected_feature} по классам",
                 labels={selected_feature: selected_feature},
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False)
    return fig

@app.callback(
    Output('bar-chart', 'figure'),
    Input('bar-feature', 'value')
)
def update_stacked_bar(selected_feature):
    # Группируем по признаку и целевой переменной
    grouped = df_analysis.groupby([selected_feature, 'income']).size().reset_index(name='count')
    
    # Считаем общее количество в каждой категории (для сортировки)
    total_counts = grouped.groupby(selected_feature)['count'].sum().sort_values(ascending=False)
    sorted_categories = total_counts.index.tolist()
    
    # Строим stacked bar chart
    fig = px.bar(
        grouped,
        x=selected_feature,
        y='count',
        color='income',
        title=f"Распределение по классам: {selected_feature}",
        labels={'count': 'Количество', selected_feature: selected_feature},
        color_discrete_map={'<=50K': '#1f77b4', '>50K': '#ff7f0e'},
        barmode='stack',
        category_orders={selected_feature: sorted_categories}  # ← ключевая строка!
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        legend_title_text='Доход',
        height=500
    )
    return fig

# --- Запуск сервера ---
if __name__ == '__main__':
    app.run(debug=True)