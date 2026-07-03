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
* **Falta de Linealidad:** El análisis exploratorio (EDA) demostró que las variables individuales tienen una correlación lineal casi nula con el fraude (< 0.05). El fraude ocurre en intersecciones complejas de comportamiento.

## 🛠️ Technical Architecture & Methodology

1. **Data Engineering & Memory Optimization:** 
   Procesamiento de 500K filas implementando técnicas de *Downcasting*, reduciendo el consumo de memoria en más del 60% para despliegues ligeros.
2. **Feature Engineering:**
   Creación de variables derivadas enfocadas en comportamiento transaccional (ej. `velocity_urgency_index`, `ratio_amount_to_monthly_spend`).
3. **Advanced Modeling (Supervised):**
   Entrenamiento de **XGBoost** ajustando la métrica PR-AUC mediante validación cruzada (`RandomizedSearchCV`). Se aplicó optimización matemática del umbral de decisión (Threshold Moving) para equilibrar el rendimiento.
4. **Explainable AI (XAI):**
   Integración de valores **SHAP** para proporcionar interpretabilidad local (auditoría de transacciones específicas) y global.

## 📈 Results & Business Impact

El modelo supervisado actúa como un **filtro de primera línea**, calibrado para proteger la experiencia del usuario:
* **Aprobación Legítima (Recall Clase 0): 88%**. Se mitigó el bloqueo erróneo masivo, salvando a la institución de altos costos operativos en el centro de atención al cliente.
* **Captura de Fraude (Recall Clase 1): 22%**. Se identifican de forma automática 1 de cada 5 fraudes históricos conocidos. 
* **Límite Teórico:** Los resultados confirman que las reglas estáticas y los modelos supervisados puros son insuficientes ante el fraude mutante, requiriendo arquitecturas híbridas.

## 📂 Project Structure

```text
fintech-fraud-detection/
├── data/               # Directorio ignorado en git (raw/ y processed/)
├── notebooks/  
│   ├── 01_EDA_NoProcessing.ipynb           
│   ├── 02_EDA_Risk_Analysis.ipynb
│   └── 03_Modeling_Baseline.ipynb  
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            
│   ├── model.py                    
│   └── explainability.py           
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
* [ ] Unsupervised Learning Layer: Integrar un motor de aprendizaje no supervisado mediante el modelado con algoritmos como KMeans para agrupar comportamientos transaccionales y detectar anomalías de "día cero" que el modelo supervisado no logra capturar.

* [ ] Implementar validación cruzada temporal (Time-Series Split) para evitar data leakage.

Este proyecto es parte de un portafolio profesional en Data Science y Análisis de Riesgo.