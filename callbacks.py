import dash
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from visualizations import plotly_tree 

def register_callbacks(app, df_analysis, silhouette_figures, prep, feature_names):
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
            hover_data=['count'],
            color='income_num',
            color_continuous_scale='Blues'
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        return fig

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
            hover_data=['count'],
            height=500
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            xaxis_title=factor1,
            yaxis_title="Доля >50K",
            legend_title=factor2
        )
        return fig

    @app.callback(
        Output('silhouette-plot', 'figure'),
        [Input('n-clusters-dropdown', 'value')]
    )
    def update_silhouette_plot(selected_n_clusters):
        if selected_n_clusters in silhouette_figures:
            return silhouette_figures[selected_n_clusters]
        else:
            return go.Figure().update_layout(title="График не найден")

    @app.callback(
        Output('dashboard', 'children'),
        Input('depth-dropdown', 'value')
    )
    def update_dashboard(depth):
        model = joblib.load(f'models/tree_depth_{depth}.joblib')

        X_test = prep['X_test']
        y_test = prep['y_test']
        X_test_pca2 = prep['X_test_pca2']
        X_test_pca3 = prep['X_test_pca3']

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_test, y_pred)

        tree_fig = plotly_tree(model, feature_names)

        main_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"ROC-кривая (AUC = {roc_auc:.3f})",
                "Матрица ошибок",
            ),
            vertical_spacing=0.15, horizontal_spacing=0.12
        )

        COLOR_GT50K = '#3498db'
        COLOR_LE50K = '#e74c3c'

        main_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                      line=dict(width=4, color='#e67e22')), row=1, col=1)
        main_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                      line=dict(dash='dash', color='gray')), row=1, col=1)

        main_fig.add_trace(go.Heatmap(z=cm, text=cm, texttemplate="%{text}",
                                      colorscale='Blues', showscale=False,
                                      x=['≤50K', '>50K'], y=['≤50K', '>50K']), row=1, col=2)

        pca2_true = px.scatter(
            x=X_test_pca2[:, 0], y=X_test_pca2[:, 1],
            color=y_test.map({0: '<=50K', 1: '>50K'}),
            labels={'x': 'PC1', 'y': 'PC2', 'color': 'Доход (факт)'},
            color_discrete_sequence=[COLOR_LE50K, COLOR_GT50K]
        )

        fig_3d = px.scatter_3d(x=X_test_pca3[:,0], y=X_test_pca3[:,1], z=X_test_pca3[:,2],
                               color=y_test.map({0:'≤50K', 1:'>50K'}),
                               color_discrete_map={'≤50K': COLOR_LE50K, '>50K': COLOR_GT50K},
                               labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3', 'color': 'Доход'})
        fig_3d.update_traces(marker=dict(size=4))
        fig_3d.update_layout(height=700, scene_camera=dict(eye=dict(x=1.7, y=1.7, z=1.3)))

        legend = html.Div([
            html.Strong("Цветовая легенда:"),
            html.Div([
                html.Span("●", style={'color': COLOR_LE50K, 'fontSize': 30, 'margin': '0 10px'}),
                html.Span("≤50K", style={'fontSize': 18}),
                html.Span(" "),
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

            html.H2("Диаграмма рассеивания главных компонент (PCA)", style={'textAlign': 'center'}),

            html.Div([
                dcc.Graph(figure=pca2_true, style={"height": "500px", "width": "600px"}),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "justifyContent": "center",
                "width": "100%"
            }),

            dcc.Graph(figure=fig_3d),
        ], style={'maxWidth': '1500px', 'margin': '0 auto'})