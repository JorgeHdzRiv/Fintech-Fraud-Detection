# FinTech Fraud Detection: Machine Learning Pipeline 🛡️💳

> **Sistema integral de aprendizaje automático para la detección de fraude en FinTech. Incluye optimización de memoria, manejo de datos desequilibrados e IA explicable (SHAP), implementado a través de una interfaz interactiva Streamlit.**

🚀 [Prueba la aplicación en vivo aquí](https://fraud-engine.streamlit.app/)

## 📖 Contexto de Negocio

En el sector FinTech, el fraude es un problema de optimización asimétrica. 
* Los **Falsos Negativos** (fraude aprobado) generan pérdidas directas.
* Los **Falsos Positivos** (clientes bloqueados) dañan el *Lifetime Value* (LTV) y saturan los centros de soporte.

Este motor de riesgo busca un equilibrio matemático, actuando como un filtro de primera línea que evalúa transacciones en tiempo real basándose en el historial y comportamiento espacial del usuario.

## 📊 Retos Técnicos y Limitaciones Descubiertas

Durante la fase de pruebas de estrés (Stress Testing) con la aplicación interactiva, se documentaron hallazgos clave sobre el comportamiento de los modelos basados en árboles:
1. **El Límite del Aprendizaje Supervisado (Underfitting por Desbalance):** Al forzar el hiperparámetro de hiper-sensibilidad (`max_depth=9`, `learning_rate=0.1`), el modelo colapsó (PR-AUC cayó a 0.02) debido a la memorización de ruido en el 2.3% de la clase minoritaria. Se optó por un modelo estable más conservador.
2. **Sesgo Categórico (Overfitting a "Bolsillos Seguros"):** Mediante la auditoría con **SHAP**, descubrimos que el modelo otorgaba "Pases VIP" a transacciones altamente anómalas simplemente porque ocurrían en categorías históricamente seguras (ej. *Fashion*) o por la alta antigüedad de la cuenta.

## 🛠️ Solución Arquitectónica y Despliegue

Para mitigar estas limitaciones en producción, se implementaron las siguientes estrategias en la interfaz operativa de Streamlit:
* **Threshold Moving (Ajuste Operativo):** Se redujo el umbral de decisión estricto a **42.0%** en la interfaz para contrarrestar el conservadurismo del modelo ante ataques de ráfaga (Card-Not-Present).
* **Ingeniería de Características en Tiempo Real:** El simulador inyecta variables temporales y espaciales (ej. *velocity_urgency_index*) recalculadas al vuelo antes de la predicción.
* **Auditoría XAI Dinámica:** Integración visual de valores Shapley para auditar por qué el motor aprueba o declina cada intento de fraude.

## 📂 Estructura del Proyecto

```text
fintech-fraud-detection/
├── data/               # raw/ y processed/ (Ignorados en git)
├── models/             # xgb_fraud_tuned.joblib (Modelo estable)
├── notebooks/    
│   ├── 01_EDA_NoProcessing.ipynb      
│   ├── 02_EDA_Risk_Analysis.ipynb
│   └── 03_Modeling_Baseline.ipynb  
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            
│   ├── explainability.py           
├── api/
│   └── app.py                      # Dashboard Interactivo Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Instalación y Ejecución Local
1. Clonar el repositorio:
```bash
git clone [https://github.com/JorgeHdzRiv/Fintech-Fraud-Detection.git](https://github.com/JorgeHdzRiv/Fintech-Fraud-Detection.git)
cd fintech-fraud-detection
```

2. Crear un entorno virtual e instalar dependencias:
```bash
python -m venv venv
source venv/bin/activate  # En Windows usa: venv\Scripts\activate
pip install -r requirements.txt
```

3. Ejecutar el dashboard interactivo de riesgo (Streamlit):
```bash
streamlit run api/app.py
```

## 📋 Next Steps / Roadmap
* [ ] Módulo de Aprendizaje No Supervisado (KMeans): Dado que el modelo supervisado es vulnerable a ataques de fraude de "Día Cero" disfrazados en categorías seguras, el siguiente paso es integrar un algoritmo KMeans para detectar anomalías espaciales y clústeres de comportamiento, operando en paralelo al XGBoost.

Este proyecto es parte de un portafolio profesional en Data Science y Análisis de Riesgo.