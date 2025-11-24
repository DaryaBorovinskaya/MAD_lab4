import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import numpy as np


def create_histograms(df_analysis, numeric_cols):
    rows = 2
    cols = 3
    count_intervals = int(1 + math.log2(df_analysis.shape[0]))
    if count_intervals <= 0:
        count_intervals = 1

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
        
        row_pos = (i // cols) + 1
        col_pos = (i % cols) + 1

        bins = np.linspace(data.min(), data.max(), count_intervals + 1)
        fig.add_trace(
            go.Histogram(
                x=data,
                xbins=dict(start=bins[0], end=bins[-1], size=(bins[-1] - bins[0]) / count_intervals),
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

    fig.update_layout(
        title={
            'text': "Распределения числовых признаков",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=700,
        width=1400,
    )
    return fig

def create_correlation_heatmap(df_analysis, numeric_cols):
    corr_data = df_analysis[numeric_cols]
    correlation_matrix = corr_data.corr(method='pearson')
    labels = numeric_cols

    fig = go.Figure(data=go.Heatmap(
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

    fig.update_layout(
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
    return fig

def create_scatter_matrix(df_analysis, numeric_cols):
    df_analysis_for_scatter = df_analysis.copy()
    df_analysis_for_scatter['income_num'] = df_analysis_for_scatter['income_num'].astype(str)
    df_analysis_for_scatter['income_label'] = df_analysis_for_scatter['income_num'].map({'0': 'доход <=50', '1': 'доход > 50'})

    fig = px.scatter_matrix(
        df_analysis_for_scatter,
        dimensions=numeric_cols,
        color='income_label',
        color_discrete_map={'доход <=50': 'lightcoral', 'доход > 50': 'lightgreen'},
        title="Матрица диаграмм рассеивания числовых признаков"
    )

    fig.update_layout(
        title={
            'text': "Матрица диаграмм рассеивания числовых признаков",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=1000,
        width=1000,
        legend=dict(
            title="Доход ",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    return fig

def create_confusion_matrix(cm, title='Матрица ошибок'):
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale='Blues',
        title=title,
        labels=dict(x="Предсказано", y="Фактически"),
        x=['<50K', '>50K'],
        y=['<50K', '>50K']
    )
    fig.update_traces(
        hovertemplate="<b>Фактически:</b> %{y}<br><b>Предсказано:</b> %{x}<br><b>Количество:</b> %{z}<extra></extra>"
    )
    fig.update_layout(
        xaxis_title="Предсказаный класс",
        yaxis_title="Фактический класс",
        height=500,
        width=500
    )
    return fig

def create_roc_curve(fpr, tpr, roc_auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC-кривая (AUC = {roc_auc:.2f})',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Случайная классификация',
        line=dict(color='red', width=1, dash='dash')
    ))
    fig.update_layout(
        title='ROC-кривая',
        xaxis_title='Доля ложных срабатываний (FPR)',
        yaxis_title='Доля истинных срабатываний (TPR)',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=500,
        width=500
    )
    return fig

def create_feature_coefficients(coefficients, feature_names):
    fig = go.Figure(data=go.Bar(
        x=feature_names,
        y=coefficients,
        marker_color='skyblue',
        marker_line_color='black',
        marker_line_width=1
    ))
    fig.update_layout(
        title='Коэффициенты признаков',
        xaxis_title='Признак',
        yaxis_title='Коэффициент',
        height=500,
        width=800
    )
    return fig

def create_silhouette_plots(cluster_results, n_clusters_range):
    def plot_silhouette_from_results(results_dict, n_clusters):
        if n_clusters not in results_dict:
            print(f"Результаты для n_clusters={n_clusters} не найдены.")
            return go.Figure().update_layout(title=f"Нет данных для {n_clusters} кластеров")

        data = results_dict[n_clusters]
        cluster_labels = data['labels']
        sample_silhouette_values = data['silhouette_samples']
        silhouette_avg = data['silhouette_avg']

        unique_labels = np.unique(cluster_labels)
        n_clusters_actual = len(unique_labels)

        fig_sil = go.Figure()

        max_points_for_plot = 1000
        total_points = len(sample_silhouette_values)

        if total_points > max_points_for_plot:
            np.random.seed(42)
            sample_indices = np.random.choice(total_points, size=max_points_for_plot, replace=False)
            cluster_labels_vis = cluster_labels[sample_indices]
            sample_silhouette_values_vis = sample_silhouette_values[sample_indices]
        else:
            cluster_labels_vis = cluster_labels
            sample_silhouette_values_vis = sample_silhouette_values

        colors = px.colors.qualitative.Set2
        y_offset = 0

        for i in range(n_clusters_actual):
            mask = cluster_labels_vis == i
            ith_cluster_silhouette_values = sample_silhouette_values_vis[mask]

            if len(ith_cluster_silhouette_values) == 0:
                continue

            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_vals = np.arange(y_offset, y_offset + size_cluster_i)

            x_vals = ith_cluster_silhouette_values.flatten()
            y_vals = y_vals.flatten()

            fig_sil.add_trace(go.Bar(
                x=x_vals,
                y=y_vals,
                orientation='h',
                name=f'Кластер {i}',
                marker_color=colors[i % len(colors)],
                marker_line=dict(width=0),
                opacity=1.0,
                showlegend=True,
            ))

            y_offset += size_cluster_i

        fig_sil.add_shape(
            type='line',
            x0=silhouette_avg, y0=0,
            x1=silhouette_avg, y1=y_offset,
            line=dict(color='red', width=2, dash='dash'),
        )
        fig_sil.add_annotation(
            x=silhouette_avg,
            y=y_offset,
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
            yaxis_title='Номер объекта (по кластерам)',
            height=500,
            width=700,
            showlegend=True
        )

        return fig_sil

    silhouette_figures = {}
    for n in n_clusters_range:
        silhouette_figures[n] = plot_silhouette_from_results(cluster_results, n)
    return silhouette_figures

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

        majority_class = 1 if value[1] > value[0] else 0
        color = "#e74c3c" if majority_class == 0 else "#3498db"

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

        if parent_x is not None:
            edge_traces.append(go.Scatter(
                x=[parent_x, x], y=[parent_y, y],
                mode="lines",
                line=dict(color="#555", width=2),
                hoverinfo="none",
                showlegend=False,
            ))

        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left != -1:
            recurse(left, x - dx, y - 1, dx * 0.65, x, y)
        if right != -1:
            recurse(right, x + dx, y - 1, dx * 0.65, x, y)

    recurse(0, x=0, y=0, dx=1.2)

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