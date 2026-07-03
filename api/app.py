import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# 1. Configuración de rutas para importar nuestros módulos locales
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# IMPORTAMOS reduce_mem_usage para arreglar el problema de los tipos de datos
from preprocessing import create_features, reduce_mem_usage
from explainability import get_shap_explainer, plot_local_explanation

# 2. Configuración de la página
st.set_page_config(page_title="Fraud Risk Engine", page_icon="🛡️", layout="wide")

# 3. Funciones de Carga con Caché para optimizar rendimiento
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_fraud_tuned.joblib')
    return joblib.load(model_path)

@st.cache_data
def load_sample_data():
    """
    Carga una muestra, aplica los tipos de datos correctos (category) 
    y retorna tanto los datos procesados para SHAP como una plantilla cruda para el simulador.
    """
    # NUEVA RUTA: Apunta directamente al archivo que acabamos de crear en la carpeta api/
    data_path = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
    df = pd.read_csv(data_path)
    
    # CORRECCIÓN: Convertimos las columnas a 'category' como espera XGBoost
    df = reduce_mem_usage(df, verbose=False)
    
    # Guardamos la primera fila como plantilla inmutable para respetar los tipos de datos
    df_raw_template = df.iloc[[0]].copy()
    
    # Aplicamos feature engineering para el dataset base de SHAP
    df_features = create_features(df)
    
    cols_to_drop = ['is_fraud', 'transaction_id', 'customer_id', 'transaction_time']
    cols_to_drop = [col for col in cols_to_drop if col in df_features.columns]
    X_sample = df_features.drop(columns=cols_to_drop)
    
    return X_sample, df_raw_template

# 4. Inicialización de recursos
st.title("🛡️ FinTech Fraud Detection Engine")
st.markdown("""
Esta herramienta evalúa el riesgo transaccional en tiempo real utilizando un modelo XGBoost optimizado. 
Integrado con **SHAP**, proporciona explicabilidad instantánea para auditoría de decisiones.
""")

with st.spinner("Cargando Motor de Riesgo..."):
    model = load_model()
    # Desempaquetamos la muestra y la plantilla
    X_sample, raw_template = load_sample_data()
    explainer, shap_values = get_shap_explainer(model, X_sample)

# 5. Panel Lateral (Simulador de Transacciones)
st.sidebar.header("Simulador de Transacción")

# NUEVO: Extraemos las categorías de la plantilla para el menú desplegable
opciones_categoria = raw_template['merchant_category'].cat.categories.tolist()

st.sidebar.markdown("### 👤 Perfil del Cliente")
sim_tenure = st.sidebar.slider("Antigüedad de cuenta (Años)", 0.0, 25.0, 0.5) 
sim_risk_score = st.sidebar.slider("Score de Riesgo (Sistema Legacy)", 0, 100, 85) 

st.sidebar.markdown("### 💳 Detalles de la Compra")
# NUEVO: Agregamos el selector de categoría
sim_category = st.sidebar.selectbox("Categoría del Comercio", options=opciones_categoria)

sim_amount = st.sidebar.number_input("Monto de la Transacción ($)", min_value=1.0, value=1800.0)
sim_dist = st.sidebar.slider("Distancia desde casa (km)", 0.0, 1000.0, 650.0)
sim_monthly = st.sidebar.number_input("Gasto Mensual Típico ($)", min_value=10.0, value=2000.0)
sim_card_present = st.sidebar.selectbox("¿Tarjeta Física Presente?", options=[0, 1], format_func=lambda x: "Sí" if x==1 else "No (Online)")
sim_gap = st.sidebar.slider("Horas desde última compra", 0.0, 72.0, 0.5)
sim_daily_count = st.sidebar.slider("Transacciones hoy", 1, 20, 12)

# 6. Panel Principal de Resultados
st.divider()

if st.sidebar.button("Evaluar Riesgo", type="primary"):
    
    input_data = raw_template.copy()
    
    # Inyectamos TODOS los datos, incluyendo la categoría
    input_data['account_tenure_years'] = sim_tenure
    input_data['risk_score'] = sim_risk_score
    input_data['merchant_category'] = sim_category # INYECCIÓN DE CATEGORÍA
    input_data['transaction_amount'] = sim_amount
    input_data['distance_from_home'] = sim_dist
    input_data['monthly_spend'] = sim_monthly
    input_data['card_present'] = sim_card_present
    input_data['previous_transaction_gap'] = sim_gap
    input_data['daily_transaction_count'] = sim_daily_count
    
    # Forzar tipos correctos
    input_data = reduce_mem_usage(input_data, verbose=False)
            
    # Realizar Ingeniería de Características (Feature Engineering) al vuelo
    input_features = create_features(input_data)
    
    # Limpiar columnas sobrantes
    cols_to_drop = ['is_fraud', 'transaction_id', 'customer_id', 'transaction_time']
    cols_to_drop = [col for col in cols_to_drop if col in input_features.columns]
    input_features = input_features.drop(columns=cols_to_drop)
    
    # Asegurar el orden de las columnas idéntico al entrenamiento
    input_features = input_features[X_sample.columns]
    
    # Predicción
    proba = model.predict_proba(input_features)[0, 1]
    umbral = 0.45
    
    # ---------------------------------------------------------
    # NUEVO DISEÑO: Veredicto arriba, SHAP a ancho completo abajo
    # ---------------------------------------------------------
    st.subheader("Veredicto de Riesgo")
    
    # Usamos columnas solo para centrar el mensaje de éxito/error y la métrica
    col_msg, col_metric = st.columns([3, 1])
    
    with col_msg:
        if proba >= umbral:
            st.error(f"🚨 TRANSACCIÓN DECLINADA (Probabilidad de Fraude: {proba:.1%})")
        else:
            st.success(f"✅ TRANSACCIÓN APROBADA (Probabilidad de Fraude: {proba:.1%})")
            
    with col_metric:
        st.metric(label="Umbral de Riesgo", value=f"{umbral*100:.1f}%")
        
    st.divider()
    
    # SHAP fuera de las columnas para que tome todo el ancho de la página
    st.subheader("Auditoría XAI (SHAP)")
    with st.spinner("Generando explicación gráfica..."):
        local_shap_values = explainer.shap_values(input_features)
        fig_local = plot_local_explanation(explainer, local_shap_values, input_features, transaction_index=0)
        
        st.pyplot(fig_local, width='stretch')