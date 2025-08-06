import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configuración de la página
st.set_page_config(
    page_title="Análisis Energético Colombia",
    page_icon="⚡",
    layout="wide"
)

# Título principal
st.title("📊 Análisis del Balance Energético Colombiano")

def load_data(uploaded_file):
    """Carga datos desde archivo CSV o Excel"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

def show_exploratory_analysis(df):
    """Muestra análisis exploratorio de datos"""
    st.header("Exploración Inicial de Datos")
    
    if st.checkbox("Mostrar datos crudos"):
        st.dataframe(df)
    
    st.subheader("Estadísticas Descriptivas")
    st.dataframe(df.describe())
    
    var_hist = st.selectbox("Seleccione variable para histograma", df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[var_hist], kde=True, ax=ax)
    st.pyplot(fig)

def show_correlation_analysis(df):
    """Muestra análisis de correlación"""
    st.header("Análisis de Correlaciones")
    
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Gráfico de Dispersión Interactivo")
    col1, col2 = st.columns(2)
    x_axis = col1.selectbox("Eje X", df.columns)
    y_axis = col2.selectbox("Eje Y", df.columns)
    fig = px.scatter(df, x=x_axis, y=y_axis, hover_data=[df.index])
    st.plotly_chart(fig, use_container_width=True)

def train_predictive_models(df):
    """Entrena y muestra modelos predictivos"""
    st.header("Modelos de Machine Learning")
    
    st.subheader("Configuración del Modelo")
    features = st.multiselect(
        "Seleccione variables predictoras",
        df.columns,
        default=df.columns[:3] if len(df.columns) >= 3 else df.columns
    )
    target = st.selectbox("Variable objetivo", df.columns)
    
    if st.button("Entrenar Modelos") and features and target:
        try:
            X = df[features]
            y = df[target]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                show_linear_regression(X_train, X_test, y_train, y_test, features)
            
            with col2:
                show_random_forest(X_train, X_test, y_train, y_test, features)
        
        except Exception as e:
            st.error(f"Error al entrenar modelos: {e}")

def show_linear_regression(X_train, X_test, y_train, y_test, features):
    """Muestra resultados de regresión lineal"""
    st.subheader("Regresión Lineal")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    st.metric("R²", f"{r2_score(y_test, y_pred):.3f}")
    st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2e}")
    
    coeff_df = pd.DataFrame({
        'Variable': features,
        'Coeficiente': lr.coef_
    }).sort_values('Coeficiente', ascending=False)
    st.dataframe(coeff_df)

def show_random_forest(X_train, X_test, y_train, y_test, features):
    """Muestra resultados de random forest"""
    st.subheader("Random Forest")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    st.metric("R²", f"{r2_score(y_test, y_pred_rf):.3f}")
    st.metric("MSE", f"{mean_squared_error(y_test, y_pred_rf):.2e}")
    
    importances = pd.DataFrame({
        'Variable': features,
        'Importancia': rf.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(x='Importancia', y='Variable', data=importances, ax=ax)
    st.pyplot(fig)

def show_clustering_analysis(df):
    """Muestra análisis de clustering"""
    st.header("Análisis de Clustering")
    
    cluster_features = st.multiselect(
        "Variables para clustering",
        df.columns,
        default=df.columns[:2] if len(df.columns) >= 2 else df.columns
    )
    
    if cluster_features and st.button("Ejecutar Clustering"):
        try:
            X = df[cluster_features]
            X_scaled = StandardScaler().fit_transform(X)
            
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_scaled)
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                principal_components[:, 0], 
                principal_components[:, 1], 
                c=clusters, 
                cmap='viridis',
                alpha=0.6
            )
            plt.colorbar(scatter)
            ax.set_xlabel('Componente Principal 1')
            ax.set_ylabel('Componente Principal 2')
            ax.set_title('Resultados de Clustering (K=3)')
            st.pyplot(fig)
            
            df_cluster = df.copy()
            df_cluster['Cluster'] = clusters
            st.dataframe(df_cluster.groupby('Cluster')[cluster_features].mean())
        
        except Exception as e:
            st.error(f"Error en clustering: {e}")

# Carga de archivos en el sidebar
st.sidebar.header("Carga de Datos")
uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo (CSV o Excel)",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.sidebar.success("Archivo cargado correctamente!")
        
        st.sidebar.header("Opciones de Análisis")
        analysis_type = st.sidebar.selectbox(
            "Seleccione el tipo de análisis",
            [
                "Exploración de Datos",
                "Correlaciones",
                "Modelos Predictivos",
                "Clustering"
            ]
        )

        if analysis_type == "Exploración de Datos":
            show_exploratory_analysis(df)
        elif analysis_type == "Correlaciones":
            show_correlation_analysis(df)
        elif analysis_type == "Modelos Predictivos":
            train_predictive_models(df)
        else:
            show_clustering_analysis(df)
    else:
        st.sidebar.error("Error al procesar el archivo. Verifica el formato.")
else:
    st.info("Por favor, sube un archivo CSV o Excel para comenzar el análisis")

# Notas finales
st.sidebar.markdown("---")
st.sidebar.info(
    "🔍 Herramienta de análisis de datos energéticos\n\n"
    "Creada con Streamlit | Python"
)
