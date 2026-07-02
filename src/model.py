import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, classification_report
import xgboost as xgb

def prepare_and_split(df: pd.DataFrame, target_col: str = 'is_fraud', test_size: float = 0.2):
    """
    Separa las variables predictoras del target y realiza un split estratificado.
    Elimina columnas que no deben entrar al modelo (IDs y fechas).
    """
    print("Preparando datos para modelado...")
    
    # Excluimos IDs, fechas y la variable objetivo
    cols_to_drop = [target_col, 'transaction_id', 'customer_id', 'transaction_time']
    # Aseguramos que solo intentamos borrar columnas que realmente existen en el df
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    print(f"Dimensiones Train: {X_train.shape}")
    print(f"Dimensiones Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_xgboost_baseline(X_train, y_train, X_test, y_test):
    """
    Entrena un modelo XGBoost ajustando el peso de las clases por el desbalanceo.
    Calcula la métrica PR-AUC.
    """
    print("Entrenando modelo XGBoost Baseline...")
    
    # Calculamos el ratio de clases negativas vs positivas para el parámetro scale_pos_weight
    neg_class_count = len(y_train[y_train == 0])
    pos_class_count = len(y_train[y_train == 1])
    scale_pos_weight = neg_class_count / pos_class_count 
    
    # Configuramos el clasificador.
    # enable_categorical=True permite que XGBoost maneje las categorías sin necesidad de One-Hot Encoding
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        enable_categorical=True,
        tree_method='hist', # Optimizado para datasets grandes y categorías
        random_state=42,
        n_estimators=100,
        max_depth=5
    )
    
    clf.fit(X_train, y_train)
    
    # Predecimos probabilidades, no clases duras (0 o 1)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # Calculamos PR-AUC (Nuestra métrica estrella para datos desbalanceados)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    print(f"\n--- Resultados Baseline ---")
    print(f"PR-AUC Score: {pr_auc:.4f}")
    
    # Predicciones duras para el reporte tradicional
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return clf