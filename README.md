# FinTech Fraud Detection: Machine Learning Pipeline 🛡️💳

> **Sistema integral de aprendizaje automático para la detección de fraude en FinTech. Incluye optimización de memoria, manejo de datos desequilibrados e IA explicable (SHAP), implementado a través de una interfaz interactiva Streamlit.**

## 📖 Business Context

En el sector FinTech, la detección de fraude es un problema de optimización asimétrica. 
* Los **Falsos Negativos** (transacciones fraudulentas aprobadas) resultan en pérdidas financieras directas y un daño a la reputación de la institución.
* Los **Falsos Positivos** (transacciones legítimas declinadas) generan fricción con el cliente, afectando la retención y el *Lifetime Value* (LTV).

Este proyecto desarrolla un motor de riesgo transaccional enfocado en maximizar la captura de fraude minimizando el impacto en clientes legítimos, utilizando datos desbalanceados e integrando interpretabilidad (XAI) para auditoría de reglas de negocio.

## 📊 The Challenge: Imbalanced Data & Non-Linear Patterns

El modelo fue entrenado con un dataset transaccional de 500K registros, presentando los siguientes retos:
* **Desbalance Extremo:** Solo el 2.34% de las transacciones son fraudulentas.
* **Falta de Linealidad:** El análisis exploratorio (EDA) demostró que las variables individuales (como el monto o la distancia) tienen una correlación lineal casi nula con el fraude (< 0.05). El fraude ocurre en intersecciones complejas de comportamiento.

## 🛠️ Technical Architecture & Methodology

1. **Data Engineering & Memory Optimization:** 
   Procesamiento de 500K filas implementando técnicas de *Downcasting* (reducción de `float64` a `float16/32` y optimización de tipos categóricos), reduciendo el consumo de memoria en más del 60% para despliegues ligeros.
2. **Feature Engineering:**
   Creación de variables derivadas enfocadas en comportamiento transaccional (ej. `velocity_urgency_index`, `ratio_amount_to_monthly_spend`) para exponer patrones anómalos a los algoritmos basados en árboles.
3. **Advanced Modeling:**
   Entrenamiento y evaluación de modelos de ensamble robustos ante el desbalanceo (**XGBoost, LightGBM**). Se descartó el uso de *Accuracy* en favor de métricas críticas para riesgo: **PR-AUC (Precision-Recall)** y **F-Beta Score**.
4. **Explainable AI (XAI):**
   Integración de valores **SHAP** para proporcionar interpretabilidad local (explicar por qué una transacción específica fue bloqueada) y global (importancia de variables en el motor de riesgo).
5. **Interactive Deployment:**
   El pipeline predictivo y los dashboards de análisis están empaquetados para ser consumidos mediante una aplicación web interactiva en **Streamlit**, diseñada para ser el panel de control de un analista de fraude.

## 📂 Project Structure

```text
fintech-fraud-detection/
├── data/               # Directorio ignorado en git por size(raw/ y processed/)
├── notebooks/          
│   └── 01_EDA_Risk_Analysis.ipynb  # Análisis exploratorio y correlaciones
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Downcasting y Feature Engineering
│   ├── model.py                    # Pipeline de entrenamiento XGBoost/LightGBM
│   └── explainability.py           # Generación de gráficos SHAP
├── api/
│   └── app.py                      # Aplicación interactiva (Streamlit)
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 How to Run Locally
1. Clonar el repositorio:
```bash
git clone [https://github.com/TU_USUARIO/fintech-fraud-detection.git](https://github.com/TU_USUARIO/fintech-fraud-detection.git)
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
* [ ] Implementar validación cruzada temporal (Time-Series Split) para evitar data leakage en los patrones de fraude a lo largo del tiempo.

* [ ] Empaquetar el pipeline en un contenedor de Docker para facilitar el despliegue en entornos cloud.

Este proyecto es parte de un portafolio profesional en Data Science y Análisis de Riesgo.