import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_shap_explainer(model, X_sample: pd.DataFrame):
    """
    Inicializa el explainer de SHAP usando un modelo basado en árboles.
    Se recomienda pasar una muestra de los datos (ej. 5000 filas) si el dataset es muy grande
    para evitar cuellos de botella en memoria durante el cálculo inicial.
    """
    print("Calculando valores SHAP... Esto puede tomar unos segundos.")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values

def plot_global_importance(shap_values, X_sample: pd.DataFrame):
    """
    Genera el gráfico de importancia global de las variables (Summary Plot).
    Muestra qué variables tienen mayor impacto general en la detección de fraude a nivel macro.
    """
    plt.figure(figsize=(10, 6))
    # plot_type="bar" nos da la importancia absoluta media para cada variable
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("Impacto Global de Variables en el Riesgo (SHAP)", fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    # Retornamos la figura actual (gcf = get current figure) para que Streamlit pueda renderizarla
    return plt.gcf()

def plot_local_explanation(explainer, shap_values, X_sample: pd.DataFrame, transaction_index: int = 0):
    """
    Explica una transacción individual (Auditoría Local).
    Genera un gráfico tipo 'Force Plot' para entender exactamente cómo cada variable empujó 
    la probabilidad de fraude hacia arriba o hacia abajo en ESA transacción específica.
    """
    # Verificamos la estructura de los expected_values (depende de la versión exacta de XGBoost/SHAP)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[0]
        
    plt.figure(figsize=(12, 4))
    shap.plots.force(
        expected_value, 
        shap_values[transaction_index], 
        X_sample.iloc[transaction_index, :],
        matplotlib=True,
        show=False
    )
    plt.title(f"Auditoría de Transacción (Índice: {transaction_index})", fontweight='bold')
    plt.tight_layout()
    
    return plt.gcf()