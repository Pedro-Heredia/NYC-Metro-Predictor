# NYC Metro Predictor

🇪🇸 [Español](#español) | 🇬🇧 [English](#english)

---

## Español

Aplicación web para predecir retrasos en el metro de Nueva York y calcular rutas óptimas entre estaciones.

Desarrollada como Trabajo de Fin de Grado en Ingeniería Informática.

**Demo en vivo:** https://nyc-metro-predictor.streamlit.app/

### Cómo funciona

Un servidor recolecta datos GTFS-Realtime de la MTA cada 5 minutos, los guarda en
PostgreSQL, y cada domingo se reentrena automáticamente un modelo XGBoost sobre todo
el histórico acumulado (17M+ registros a fecha de esta actualización). El modelo
predice retrasos tanto para trenes ya en marcha (delay conocido) como para consultas
antes de iniciar el viaje (delay desconocido), y el enrutamiento entre estaciones se
calcula con Dijkstra sobre el grafo de la red.

Este repositorio incluye una versión reducida y autocontenida (modelo ya entrenado +
una muestra del dataset + GTFS estático) para poder ejecutar la app sin necesidad de
levantar el servidor de recolección ni la base de datos.

### Versión antigua (entrega académica)

La versión exacta que se entregó y evaluó en la universidad (mayo 2026) está conservada
en la rama [`version-academica`](../../tree/version-academica). Tenía un alcance más
reducido: 9 líneas de metro en vez de las 22 actuales, y un modelo sin el modo pre-viaje.

**Continuación:** tras la entrega, se ha seguido recopilando datos en un servidor propio
de forma continua, ampliando el alcance a las 22 líneas, añadiendo el modo de predicción
pre-viaje, y automatizando el reentrenamiento semanal del modelo sobre el histórico
acumulado. Esta rama (`main`) refleja ese desarrollo continuado.

### Tecnologías
- **Modelo:** XGBoost entrenado con datos históricos reales de la MTA, reentrenado semanalmente
- **Enrutamiento:** Dijkstra sobre grafo GTFS
- **Backend de recolección:** Python + PostgreSQL + systemd (servidor Linux propio)
- **Interfaz:** Streamlit

### Estructura del repositorio
- `app.py`, `predictor.py`, `service_types.py` — aplicación Streamlit (lo que corre en la demo)
- `modelo_xgboost_final.pkl`, `dataset_viajes_raw.pkl` — modelo entrenado y una muestra del dataset
- `stops.txt`, `stop_times.txt`, `trips.txt`, `routes.txt` — GTFS estático de la MTA
- `codigo/` — pipeline completo de producción: recolección (`collector.py`), construcción
  del dataset (`prepare_dataset.py`) y entrenamiento (`train_xgboost.py`). Ver
  `codigo/compila.txt` para replicarlo desde cero.

### Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Autor
Pedro Heredia Torres — Universidad de Sevilla, 2026

---

## English

Web app that predicts NYC subway delays and calculates optimal routes between stations.

Built as a final degree project (Computer Engineering).

**Live demo:** https://nyc-metro-predictor.streamlit.app/

### How it works

A server collects GTFS-Realtime data from the MTA every 5 minutes, stores it in
PostgreSQL, and every Sunday automatically retrains an XGBoost model on the full
accumulated history (17M+ records as of this update). The model predicts delays both
for trains already en route (known delay) and for pre-trip queries (unknown delay),
and routing between stations is computed with Dijkstra over the network graph.

This repository includes a reduced, self-contained version (a pre-trained model + a
sample of the dataset + static GTFS) so the app can run without needing to run the
collection server or the database.

### Older version (academic submission)

The exact version submitted and graded at university (May 2026) is preserved on the
[`version-academica`](../../tree/version-academica) branch. It had a smaller scope:
9 subway lines instead of the current 22, and a model without the pre-trip mode.

**Continuation:** after the submission, data collection continued on a dedicated
server, expanding coverage to all 22 lines, adding the pre-trip prediction mode, and
automating weekly model retraining on the accumulated history. This branch (`main`)
reflects that continued development.

### Tech stack
- **Model:** XGBoost trained on real MTA historical data, retrained weekly
- **Routing:** Dijkstra over the GTFS graph
- **Collection backend:** Python + PostgreSQL + systemd (self-hosted Linux server)
- **UI:** Streamlit

### Repository structure
- `app.py`, `predictor.py`, `service_types.py` — Streamlit app (what runs the demo)
- `modelo_xgboost_final.pkl`, `dataset_viajes_raw.pkl` — trained model and a dataset sample
- `stops.txt`, `stop_times.txt`, `trips.txt`, `routes.txt` — MTA static GTFS
- `codigo/` — full production pipeline: collection (`collector.py`), dataset build
  (`prepare_dataset.py`) and training (`train_xgboost.py`). See `codigo/compila.txt`
  to replicate it from scratch.

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Author
Pedro Heredia Torres — University of Seville, 2026
