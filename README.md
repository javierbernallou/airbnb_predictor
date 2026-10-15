# Airbnb Predictor

Proyecto para analizar alojamientos de Airbnb en Londres y detectar oportunidades de alquiler con precio por debajo del valor estimado por un modelo predictivo.

La aplicación combina un pipeline de preparación de datos, un modelo de regresión basado en XGBoost y una interfaz web con Streamlit para explorar chollos por barrio, tipo de vivienda y nivel de equipamiento.

## Descripción general

Este repositorio permite:

- cargar datos de Airbnb en Londres,
- limpiar y enriquecer el dataset con variables como distancia al centro, amenities, tipo de alojamiento y barrio,
- entrenar un modelo para estimar un precio "justo",
- comparar precio real frente a predicción,
- visualizar alojamientos infravalorados y exportarlos como CSV.

## Características principales

- Pipeline de procesamiento de datos en `build_dataset.py`
- Modelo predictivo con `XGBoost`
- Validación cruzada con KFold para evitar fugas de información
- Interfaz web desarrollada con Streamlit
- Mapa geográfico con Plotly / PyDeck
- Ranking de barrios, métricas y filtros interactivos
- Exportación de resultados de "chollos"

## Estructura del proyecto

```text
airbnb_predictor/
├── app.py                     # Aplicación principal en Streamlit
├── build_dataset.py           # Genera el dataset procesado y el modelo
├── analisis_inicial.ipynb     # Notebook con análisis exploratorio
├── requirements.txt           # Dependencias del proyecto
├── data/
│   ├── raw/
│   │   ├── listings.csv.gz
│   │   └── ward-profiles-excel-version.csv
│   └── processed/
│       ├── listings_scored.csv.gz
│       ├── chollos_airbnb.csv
│       └── ...
├── .gitignore
└── README.md
```

## Stack tecnológico

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- Streamlit
- Plotly
- PyDeck

## Requisitos

Antes de ejecutar el proyecto, asegúrate de tener instalado Python 3.8 o superior.

## Instalación

```bash
git clone https://github.com/javierbernallou/airbnb_predictor.git
cd airbnb_predictor

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

## Preparación de datos

El pipeline principal está en `build_dataset.py`.

Este script:

- carga el CSV raw de Airbnb,
- limpia y normaliza columnas,
- calcula variables útiles para el precio,
- genera `pred_price`,
- guarda el resultado final en `data/processed/listings_scored.csv.gz`.

Ejecuta:

```bash
python build_dataset.py
```

Si el archivo procesado no existe, la aplicación puede mostrar datos sintéticos de demostración para que puedas probar la interfaz.

## Ejecución de la aplicación

```bash
streamlit run app.py
```

La app abrirá una interfaz con:

- mapa de alojamientos,
- métricas de precios,
- análisis por barrio,
- comparación real vs predicho,
- filtros por inquilinos, precio, valoración y reseñas,
- sección de "chollos" para exportar resultados.

## Qué muestra la aplicación

La interfaz clasifica cada alojamiento con una etiqueta basada en la diferencia entre el precio real y el precio previsto:

- Infravalorado
- Precio justo
- Sobrevalorado

La lógica principal es:

```python
valoracion = precio_real / precio_predicho - 1
```

Cuando la diferencia es negativa, el alojamiento está por debajo de lo que el modelo considera justo.

## Notas del modelo

- La variable objetivo se modela en escala logarítmica para estabilizar la varianza.
- El precio estimado se genera con validación cruzada (`KFold` con 5 folds).
- Se utilizan variables como:
  - ubicación,
  - número de huéspedes permitidos,
  - número de habitaciones,
  - distancia al centro de Londres,
  - amenities,
  - revisiones y puntuaciones,
  - tipo de vivienda y barrio.

## Datos incluidos

El repositorio incluye archivos bajo `data/raw` y `data/processed` para ejecutar o probar el proyecto.

## Mejoras futuras

- Añadir tests automáticos
- Añadir Dockerfile para despliegue reproducible
- Mejorar la gestión de dependencias y versiones



