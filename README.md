# NYC Metro Predictor

Aplicación web para predecir retrasos en el metro de Nueva York y calcular rutas óptimas entre estaciones.

Desarrollada como Trabajo de Fin de Grado en Ingeniería Informática.

## Tecnologías
- **Modelo:** XGBoost entrenado con datos históricos de la MTA
- **Enrutamiento:** Dijkstra sobre grafo GTFS
- **Interfaz:** Streamlit

## Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Autor
Pedro Heredia Torres — Universidad de Sevilla, 2026
