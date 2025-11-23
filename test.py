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
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')


file_path = './adult 3 (1).csv'
df = pd.read_csv(file_path)
df = df.replace('?', pd.NA).dropna()


numeric_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
df_analysis = df.copy()
df_analysis['income_num'] = (df_analysis['income'] == '>50K').astype(int)
categorical_cols = ['workclass', 'education', 'marital-status', 
                    'occupation', 'relationship', 'race', 'gender', 
                    'native-country']
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


# --- Логистическая регрессия и визуализации ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Подготовка данных для моделирования
X = df_analysis[numeric_cols]
y = df_analysis['income_num'] # Уже числовой: 0 или 1

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Создание и обучение модели
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)


fig_pdp_list = []

for i, feature in enumerate(numeric_cols):
    # Создаём сетку значений для текущего признака
    # Берём минимум и максимум из тестовой выборки для реалистичности
    feature_min = X_test[feature].min()
    feature_max = X_test[feature].max()
    feature_range = np.linspace(feature_min, feature_max, 100)

    # Создаём массив для предсказания
    # Заполняем его средними значениями всех признаков
    X_temp = np.tile(X_test.mean().values, (len(feature_range), 1)) # Средние значения
    # Заменяем только текущий признак
    X_temp[:, i] = feature_range

    # Предсказываем вероятности
    probas = log_reg.predict_proba(X_temp)[:, 1]

    # Создаём график
    fig_pdp = go.Figure()
    fig_pdp.add_trace(go.Scatter(
        x=feature_range,
        y=probas,
        mode='lines',
        name=f'{feature}',
        line=dict(color='blue', width=2)
    ))
    fig_pdp.update_layout(
        title=f'Влияние признака "{feature}" на вероятность дохода >50K',
        xaxis_title=feature,
        yaxis_title='P(income >50K)',
        yaxis=dict(range=[0, 1]),
        height=400,
        width=600
    )
    fig_pdp_list.append(fig_pdp)



# Предсказания
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1] # Вероятности для положительного класса (1)

# --- Формирование строки уравнения ---
# Извлекаем коэффициенты и перехват
intercept = log_reg.intercept_[0]
coeffs = log_reg.coef_[0]
feature_names = numeric_cols

# Формируем строку уравнения z
# Начинаем с перехвата (intercept)
z_equation_str = f"z = {intercept:.4f}"
# Добавляем коэффициенты * признаки
for feat, coef in zip(feature_names, coeffs):
    sign = "+" if coef >= 0 else "-" # Определяем знак
    z_equation_str += f" {sign} ({abs(coef):.4f} * {feat})"

# Полное уравнение вероятности
full_equation_str = f"P(income >50K | X) = 1 / (1 + exp(-({z_equation_str})))"




# 1. Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
fig_cm = px.imshow(
    cm,
    text_auto=True,
    color_continuous_scale='Blues',
    title='Матрица ошибок (Logistic Regression)',
    labels=dict(x="Предсказано", y="Фактически"),
    x=['<50K', '>50K'],
    y=['<50K', '>50K']
)
fig_cm.update_layout(
    xaxis_title="Предсказанный класс",
    yaxis_title="Фактический класс",
    height=500,
    width=500
)

# 2. ROC-кривая
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=fpr,
    y=tpr,
    mode='lines',
    name=f'ROC-кривая (AUC = {roc_auc:.2f})',
    line=dict(color='blue', width=2)
))
fig_roc.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Случайная классификация',
    line=dict(color='red', width=1, dash='dash')
))
fig_roc.update_layout(
    title='ROC-кривая (Logistic Regression)',
    xaxis_title='Доля ложных срабатываний (FPR)',
    yaxis_title='Доля истинных срабатываний (TPR)',
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
    height=500,
    width=500
)

# 3. Важность признаков (коэффициенты)
coefficients = log_reg.coef_[0] # Извлекаем коэффициенты
feature_names = numeric_cols

fig_coef = go.Figure(data=go.Bar(
    x=feature_names,
    y=coefficients,
    marker_color='skyblue',
    marker_line_color='black',
    marker_line_width=1
))
fig_coef.update_layout(
    title='Коэффициенты признаков (Logistic Regression)',
    xaxis_title='Признак',
    yaxis_title='Коэффициент',
    height=500,
    width=800
)

pdp_figures = {}

for i, feature in enumerate(numeric_cols):
    # Создаём сетку значений для текущего признака
    feature_min = X_test[feature].min()
    feature_max = X_test[feature].max()
    feature_range = np.linspace(feature_min, feature_max, 100)

    # Создаём массив для предсказания
    X_temp = np.tile(X_test.mean().values, (len(feature_range), 1))
    X_temp[:, i] = feature_range

    # Предсказываем вероятности
    probas = log_reg.predict_proba(X_temp)[:, 1]

    # Создаём график
    fig_pdp = go.Figure()
    fig_pdp.add_trace(go.Scatter(
        x=feature_range,
        y=probas,
        mode='lines',
        name=f'{feature}',
        line=dict(color='blue', width=2)
    ))
    fig_pdp.update_layout(
        title=f'Влияние признака "{feature}" на вероятность дохода >50K',
        xaxis_title=feature,
        yaxis_title='P(income >50K)',
        yaxis=dict(range=[0, 1]),
        height=500,
        width=700
    )
    pdp_figures[feature] = fig_pdp


df_cluster = df_analysis.copy()

# Удаляем целевую переменную
df_cluster = df_cluster.drop(columns=['income', 'income_num'])

# Выбираем только числовые признаки (можно добавить и категориальные, закодировав их)
X_cluster = df_cluster[numeric_cols]

# Стандартизация признаков (важно для K-Means)
scaler = RobustScaler() 
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Преобразуем обратно в DataFrame для удобства (опционально)
X_cluster_scaled = pd.DataFrame(X_cluster_scaled, columns=numeric_cols)

# --- График силуэта ---
def plot_silhouette_for_n_clusters(X, n_clusters_range):
    all_figures = {}
    print('begin KMeans')
    for n_clusters in n_clusters_range:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)

        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        # Создаём подграфик для каждого числа кластеров
        fig_sil = go.Figure()

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            fig_sil.add_trace(go.Scatter(
                x=ith_cluster_silhouette_values,
                y=np.arange(y_lower, y_upper),
                mode='lines',
                line=dict(width=0.5),
                showlegend=False,
                hovertemplate=f'Кластер {i}<br>Ширина: %{{x}}<extra></extra>'
            ))

            y_lower = y_upper + 10  # Разрыв между кластерами

        # Горизонтальная линия для среднего значения силуэта
        fig_sil.add_shape(
            type='line',
            x0=silhouette_avg, y0=0,
            x1=silhouette_avg, y1=y_lower - 10,
            line=dict(color='red', width=2, dash='dash'),
        )
        fig_sil.add_annotation(
            x=silhouette_avg,
            y=y_lower - 10, # Приблизительно внизу графика
            text=f'Средний: {silhouette_avg:.3f}',
            showarrow=False,
            yshift=10,
            bgcolor='white',
            bordercolor='red',
            borderwidth=1
        )

        fig_sil.update_layout(
            title=f'Диаграмма силуэта для {n_clusters} кластеров',
            xaxis_title='Коэффициент силуэта',
            yaxis_title='Номер кластера',
            yaxis=dict(range=[0, y_lower]), # Устанавливаем диапазон оси Y
            height=500,
            width=700
        )

        # Сохраняем фигуру в словарь
        all_figures[n_clusters] = fig_sil
        print(f'KMeans {n_clusters}')
    print('end KMeans')
    return all_figures

# Диапазон числа кластеров для проверки
n_clusters_range = range(2, 6) # Например, от 2 до 5

silhouette_figures = plot_silhouette_for_n_clusters(X_cluster_scaled, n_clusters_range)



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

    html.H2("Модель: Логистическая регрессия", style={'textAlign': 'center'}),

    # Вывод уравнения
    html.H3("Уравнение модели:", style={'textAlign': 'center'}),
    html.Div([
        html.Pre(full_equation_str, style={
            'textAlign': 'left', # Выравнивание по левому краю внутри Div
            'margin': '20px auto', # Отступы сверху/снизу и авто-отступы по бокам для центрирования
            'padding': '10px',
            'backgroundColor': '#f4f4f4', # Лёгкий фон для выделения
            'border': '1px solid #ccc',
            'borderRadius': '5px',
            'maxWidth': '1000px', # Ограничиваем ширину
            'overflowX': 'auto', # Прокрутка, если строка длинная
            'fontFamily': 'monospace', # Моноширинный шрифт
            'fontSize': '14px'
        })
    ], style={'textAlign': 'center'}), # Центрируем родительский Div

    
    

    # Матрица ошибок
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

    # ROC-кривая
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

    # Коэффициенты признаков
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










    html.H2("Кластеризация", style={'textAlign': 'center'}),
    html.H2("Метрика силуэта", style={'textAlign': 'center'}),

    # Dropdown для выбора числа кластеров
    html.Div([
        html.Label("Число кластеров:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='n-clusters-dropdown',
            options=[{'label': str(n), 'value': n} for n in n_clusters_range],
            value=n_clusters_range[0], # Значение по умолчанию
            searchable=False,
            clearable=False,
            style={'width': '100px'} # Меньше ширина для числа
        )
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '20px'}),

    # График силуэта, который будет обновляться
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



# --- Callback для обновления графика силуэта ---
@app.callback(
    Output('silhouette-plot', 'figure'),
    [Input('n-clusters-dropdown', 'value')]
)
def update_silhouette_plot(selected_n_clusters):
    # Возвращаем график из словаря, соответствующий выбранному числу кластеров
    # Убедитесь, что selected_n_clusters есть в словаре
    if selected_n_clusters in silhouette_figures:
        return silhouette_figures[selected_n_clusters]
    else:
        # Возвращаем пустой график или график с сообщением об ошибке
        # На случай, если вдруг передадут значение не из диапазона
        return go.Figure().update_layout(title="График не найден")


if __name__ == '__main__':
   app.run(debug=True)