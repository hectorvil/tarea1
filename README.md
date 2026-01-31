# Pronóstico de demanda mensual (producto–tienda) con LightGBM

Este repositorio contiene un pipeline completo para pronosticar la **demanda mensual** de productos por tienda usando el dataset de 1C Company (Kaggle: *Predict Future Sales*). El objetivo es apoyar decisiones de inventario: **cuánto pedir y dónde colocarlo con anticipación**, buscando reducir sobrestock (inventario inmovilizado) y disminuir quiebres de stock (ventas perdidas), además de acelerar un proceso que suele depender de promedios móviles y ajustes manuales.

El enfoque principal utiliza un modelo en **dos etapas**: primero estima si habrá venta (clasificación) y después estima cuántas unidades se venderán (regresión). Esto es especialmente útil cuando la demanda es **esporádica e intermitente**, con muchos ceros.

## Resultados (alto nivel)
- Se cumple la meta de error agregada: el modelo alcanza **RMSE ≈ 1** (menor al umbral de 5 unidades).
- Aun así, el análisis por segmentos muestra que el modelo puede **subestimar picos de demanda**, que son pocos pero de alto impacto (posibles productos críticos o temporadas pico). Por ello, se recomienda complementar con criterio de negocio y/o ajustes de inventario de seguridad.

---

## Estructura del repositorio

### Notebook 1 — `01_eda_entendimiento.ipynb`
Exploración y entendimiento del problema:
- Tamaño del dataset, tipos de variables y revisión de nulos
- Distribuciones de `item_price` y `item_cnt_day` (incluyendo devoluciones)
- Transformación conceptual de ventas diarias a demanda mensual
- Estacionalidad (heatmap Año × Mes)
- Intermitencia y “series muertas” (meses con venta y recency)
- Evidencia para justificar lags y variables de “estado” de la serie

### Notebook 2 — `02_feature_engineering_intermediate.ipynb`
Tratamiento y feature engineering + guardado de base intermedia:
- Tratamiento mínimo de valores especiales (precios inválidos y outliers extremos)
- Agregación mensual del target (`item_cnt_month`) y clipping a [0, 20]
- Panel denso para los pares del test (captura ceros reales)
- Variables de series de tiempo:
  - lags (1–6 y 12), recency, intermitencia (rate_6)
  - ventanas recientes (sum/mean/std/max), estacionalidad (mes y sin/cos)
  - señales robustas de precio (log-price y “último precio conocido”)
- Agregados laggeados (global / por item / por shop) con shift para evitar leakage

Salida en `intermediate_data/`:
- `train.parquet`, `valid.parquet`, `test_features.parquet`
- `test_pairs.parquet` (shop_id, item_id)
- `meta.json` (lista de features y configuración)

### Notebook 3 — `03_modelo_final_hurdle.ipynb`
Modelación y submission:
- Carga de `intermediate_data/`
- Modelo final en dos etapas (hurdle):
  - Clasificación: venta vs no venta
  - Regresión: unidades condicionadas a venta
- Predicción final y `submission.csv`
- Guardado del modelo en `model_tarea1.pkl`

### Notebook 4 — `04_evaluacion_comparacion_simulacion.ipynb`
Evaluación y lectura operativa:
- Calibración por deciles, residuales y análisis de cola (y≥5/10/15)
- Comparación contra un baseline simple (solo regresión)
- Simulación operativa de inventario (sobrestock vs stockouts)
- Sensibilidad al costo de stockout para entender trade-offs

---

## Cómo correr el proyecto

1. Instala dependencias:
```bash
pip install pandas numpy matplotlib lightgbm pyarrow joblib

Referencias

Manokhin, V. (n.d.). Mastering modern time series forecasting: A comprehensive guide to statistical, machine learning, and deep learning models in Python (Early Access). Leanpub.

OpenAI. (2023). ChatGPT (Mar 14 version) [Large language model versión 5.2]. https://chat.openai.com/
