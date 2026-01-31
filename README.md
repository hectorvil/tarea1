# tarea1
Este repositorio contiene un pipeline completo para pronosticar la demanda mensual de productos por tienda usando el dataset de 1C Company (Kaggle: Predict Future Sales). El objetivo es apoyar decisiones de inventario: cuánto pedir y dónde colocarlo con anticipación, buscando reducir sobrestock (inventario inmovilizado) y disminuir quiebres de stock (ventas perdidas), además de acelerar un proceso que suele depender de promedios móviles y ajustes manuales.

El enfoque principal utiliza un modelo en dos etapas: primero estima si habrá venta (clasificación) y después estima cuántas unidades se venderán (regresión). Esto es especialmente útil cuando la demanda es esporádica e intermitente, con muchos ceros.

Resultados principales (alto nivel):

Se cumple la meta de error agregada: el modelo alcanza RMSE ≈ 1 (menor al umbral de 5 unidades).

Aun así, el análisis por segmentos muestra que el modelo puede subestimar picos de demanda, que son pocos pero de alto impacto (posibles productos críticos o temporadas pico). Por ello, se recomienda complementar con criterio de negocio y/o ajustes de inventario de seguridad.

Estructura del repositorio
Notebook 1 — 01_eda_entendimiento.ipynb

En este notebook se exploran los datos y se documentan hallazgos clave:

Dimensiones del dataset, tipos de variables y revisión de nulos.

Distribuciones de item_price y item_cnt_day (incluyendo devoluciones con ventas negativas).

Transformación conceptual de diario a mensual (creación de item_cnt_month).

Evidencia de estacionalidad (heatmap Año × Mes).

Evidencia de intermitencia y “series muertas” (meses con venta y recency).

Gráficas que motivan el uso de lags y variables de estado.

Objetivo: entender el problema y justificar el tratamiento posterior.

Notebook 2 — 02_feature_engineering_intermediate.ipynb

Aquí se construye la base final de modelación y se guarda como datos intermedios:

Tratamiento mínimo de valores especiales (precios inválidos y outliers extremos).

Agregación mensual del target (item_cnt_month) y clipping al rango [0, 20].

Construcción de un panel denso para los pares del test (para capturar ceros reales).

Creación de variables de series de tiempo y “estado”:

lags (1–6 y 12), recency, intermitencia (rate_6), ventanas recientes (sum/mean/std/max),

señales de estacionalidad (mes y representación cíclica),

señales robustas de precio (log-price y “último precio conocido”).

Agregados laggeados (global / por item / por shop) con shift para evitar leakage.

Salida: se guardan archivos en intermediate_data/:

train.parquet, valid.parquet, test_features.parquet

test_pairs.parquet (shop_id, item_id)

meta.json (lista de features y meses)

Notebook 3 — 03_modelo_final_hurdle.ipynb

Entrenamiento del modelo final en dos etapas y generación del submission:

Carga de intermediate_data/.

Etapa 1: clasificador que estima si habrá venta (venta vs no venta).

Etapa 2: regresor que estima unidades solo en observaciones con venta.

Predicción final combinando ambos componentes y recorte al rango [0, 20].

Generación de submission.csv.

Guardado del modelo final en model_tarea1.pkl.

Objetivo: entrenar y dejar un artefacto reproducible para predicción.

Notebook 4 — 04_evaluacion_comparacion_simulacion.ipynb

Evaluación y lectura operativa del modelo:

Diagnóstico del error: calibración por deciles, residuales y subestimación en cola (y≥5/10/15).

Entrenamiento de un baseline simple (solo regresión) para comparación.

Comparación visual entre baseline y modelo final.

Simulación operativa (inventario) para ver el trade-off entre sobrestock y stockouts.

Análisis de sensibilidad al costo de stockout para entender cuándo conviene cada enfoque.

Objetivo: convertir métricas de ML en implicaciones de negocio y riesgos.

Cómo correr el proyecto

Instala dependencias (ejemplo con pip):

pip install pandas numpy matplotlib lightgbm pyarrow joblib


Ejecuta notebooks en orden:

01_eda_entendimiento.ipynb

02_feature_engineering_intermediate.ipynb

03_modelo_final_hurdle.ipynb

04_evaluacion_comparacion_simulacion.ipynb

Notas sobre limitaciones y uso práctico

Aunque el RMSE agregado es bajo, el modelo puede subestimar picos de demanda (pocos casos, pero potencialmente críticos). En operación real, esos picos pueden corresponder a productos que no conviene dejar sin stock (por ventas, margen, o impacto en experiencia). Por eso, es recomendable:

usar reglas de negocio para SKUs prioritarios,

aplicar inventario de seguridad en temporadas pico,

y complementar con simulaciones de costo.

Referencias

Manokhin, V. (n.d.). Mastering modern time series forecasting: A comprehensive guide to statistical, machine learning, and deep learning models in Python (Early Access). Leanpub.

OpenAI. (2023). ChatGPT (Mar 14 version) [Large language model versión 5.2]. https://chat.openai.com/
