# NYC Metro Predictor

Aplicación web para predecir retrasos en el metro de Nueva York y calcular rutas óptimas entre estaciones.

Desarrollada como Trabajo de Fin de Grado en Ingeniería Informática.

**Demo en vivo:** https://damptzhv4dxdncxqbur2qb.streamlit.app/

## Cómo funciona

Un servidor recolecta datos GTFS-Realtime de la MTA cada 5 minutos, los guarda en
PostgreSQL, y cada domingo se reentrena automáticamente un modelo XGBoost sobre todo
el histórico acumulado (17M+ registros a fecha de esta actualización). El modelo
predice retrasos tanto para trenes ya en marcha (delay conocido) como para consultas
antes de iniciar el viaje (delay desconocido), y el enrutamiento entre estaciones se
calcula con Dijkstra sobre el grafo de la red.

Este repositorio incluye una versión reducida y autocontenida (modelo ya entrenado +
una muestra del dataset + GTFS estático) para poder ejecutar la app sin necesidad de
levantar el servidor de recolección ni la base de datos.

## Tecnologías
- **Modelo:** XGBoost entrenado con datos históricos reales de la MTA, reentrenado semanalmente
- **Enrutamiento:** Dijkstra sobre grafo GTFS
- **Backend de recolección:** Python + PostgreSQL + systemd (servidor Linux propio)
- **Interfaz:** Streamlit

## Estructura del repositorio
- `app.py`, `predictor.py`, `service_types.py` — aplicación Streamlit (lo que corre en la demo)
- `modelo_xgboost_final.pkl`, `dataset_viajes_raw.pkl` — modelo entrenado y una muestra del dataset
- `stops.txt`, `stop_times.txt`, `trips.txt`, `routes.txt` — GTFS estático de la MTA
- `codigo/` — pipeline completo de producción: recolección (`collector.py`), construcción
  del dataset (`prepare_dataset.py`) y entrenamiento (`train_xgboost.py`). Ver
  `codigo/compila.txt` para replicarlo desde cero.

## Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Autor
Pedro Heredia Torres — Universidad de Sevilla, 2026
