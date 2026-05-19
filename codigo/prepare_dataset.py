# Preparador del Dataset (pkl) para el EDA


import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from service_types import get_stop_service_type
warnings.filterwarnings('ignore')

# Configuracion de base de datos
DB_CONFIG = {
    "host": "localhost",
    "dbname": "delays2",
    "user": "postgres",
    "password": "tfg"
}

try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("Conexion exitosa")
except Exception as e:
    print(f"Error de conexion: {e}")
    exit(1)

# Query para extraer datos
# Solo extraemos registros con valores completos (sin NULLs)
query = """
    SELECT 
        train_id,
        route_id,
        stop_id,
        stop_name,
        estimated_time,
        scheduled_time,
        delay_minutes,
        cumulative_delay,
        hour,
        dow,
        is_peak_hour,
        timestamp
    FROM delays
    WHERE 
        scheduled_time IS NOT NULL
        AND estimated_time IS NOT NULL
        AND delay_minutes IS NOT NULL
        AND cumulative_delay IS NOT NULL
    ORDER BY train_id, timestamp
"""

df = pd.read_sql(query, conn)
conn.close()

print(f"Datos extraidos, Nº de registros: {len(df)} ")

# Construir segmentos de viaje

# Agrupar por train_id para construir viajes completos
train_groups = df.groupby('train_id')
viajes = []

total_trains = len(train_groups)
processed = 0
skipped_time_issues = 0
skipped_too_short = 0

for train_id, train_data in train_groups:
    processed += 1
    
    # Mostrar progreso cada 500 trenes
    if processed % 500 == 0:
        print(f"Procesando... {processed}/{total_trains} trenes ({processed/total_trains*100:.1f}%)")
    
    # Ordenar por timestamp
    train_data = train_data.sort_values('timestamp').reset_index(drop=True)
    
    #Como necesitamos al menos 2 paradas para crear un segmento:
    if len(train_data) < 2:
        skipped_too_short += 1
        continue
    
    # Convertir estimated_time a datetime para poder calcular diferencias
    try:
        train_data['estimated_time'] = pd.to_datetime(
            train_data['estimated_time'], 
            format='%H:%M:%S',
            errors='coerce'
        )
    except:
        train_data['estimated_time'] = pd.to_datetime(
            train_data['estimated_time'], 
            errors='coerce'
        )
    
    # Eliminar NaT (Not a Time)
    train_data = train_data.dropna(subset=['estimated_time'])#eliminamos si tiene nulls
    
    if len(train_data) < 2:
        skipped_too_short += 1
        continue
    
    # Verificar que los tiempos sean monotonicos (siempre crecientes)
    # Si no lo son, el tren tiene datos inconsistentes
    if not train_data['estimated_time'].is_monotonic_increasing:
        skipped_time_issues += 1
        continue
    
    # Calcular features de contexto historico del viaje
    train_data['delay_prev_1'] = train_data['delay_minutes'].shift(1) # Obtener el delay de la parada ANTERIOR (prev_1) y ANTE-ANTERIOR (prev_2)
    train_data['delay_prev_2'] = train_data['delay_minutes'].shift(2)
    train_data['delay_mean_prev'] = train_data['delay_minutes'].expanding().mean().shift(1)
    train_data['delay_std_prev'] = train_data['delay_minutes'].expanding().std().shift(1)
    
    # Crear segmentos entre paradas consecutivas
    for i in range(len(train_data) - 1):
        origen = train_data.iloc[i] #fila i
        destino = train_data.iloc[i + 1]
        
        # Calcular tiempo de viaje entre paradas
        time_diff = (destino['estimated_time'] - origen['estimated_time']).total_seconds() / 60
        
        # Filtrar segmentos fisicamente imposibles (menos de 30 segundos) 
        if time_diff < 0.5:
            continue
        
        # Verificar que no sea la misma parada
        if origen['stop_id'] == destino['stop_id']:
            continue
        
        # Features temporales basicas
        hour = origen['hour']
        minute = (origen['estimated_time'].hour * 60 + origen['estimated_time'].minute) % 1440
        
        # Features de hora pico granulares
        is_morning_rush = 1 if 7 <= hour <= 9 else 0
        is_lunch_rush = 1 if 13 <= hour <= 15 else 0
        is_evening_rush = 1 if 17 <= hour <= 19 else 0
        minute_of_hour = origen['estimated_time'].minute
        is_night = 1 if (hour >= 22 or hour < 6) else 0
        
        # Features de posicion en el viaje (feature adicional, mejor precision mejor mae)
        segment_number = i + 1
        total_segments = len(train_data) - 1
        position_ratio = segment_number / total_segments if total_segments > 0 else 0.5
        is_early_segment = 1 if position_ratio < 0.33 else 0
        is_late_segment = 1 if position_ratio > 0.67 else 0
        
        # Features de contexto historico
        delay_prev_1 = origen['delay_prev_1'] if origen['delay_prev_1'] is not None else 0
        delay_prev_2 = origen['delay_prev_2'] if origen['delay_prev_2'] is not None else 0
        delay_mean_prev = origen['delay_mean_prev'] if not pd.isna(origen['delay_mean_prev']) else 0
        delay_std_prev = origen['delay_std_prev'] if not pd.isna(origen['delay_std_prev']) else 0.5
        
        # Features de linea (las lineas 3, 5, 7 son express oficialmente segun MTA)
        #(ademas comprobado por anteriores EDAs que las lineas 3 y 5 son las mas problematicas,
        #son las que mas se retrasan)
        route_id = origen['route_id']
        is_line_3 = 1 if route_id == '3' else 0
        is_line_5 = 1 if route_id == '5' else 0
        is_express = 1 if route_id in ['3', '5', '7'] else 0
        
        # Feature de aceleracion del delay
        delay_change = destino['delay_minutes'] - origen['delay_minutes']
        delay_acceleration = abs(delay_change) / time_diff if time_diff > 0 else 0
        # para medir que tan rapido aumenta o disminuye el retraso
        
        # Features de interaccion
        hour_x_position = hour * position_ratio
        rush_x_segment = (is_morning_rush + is_evening_rush) * segment_number
        
        # Crear registro del segmento
        viaje = {
            # Identificadores
            'train_id': train_id,
            'route_id': route_id,
            'segment_number': segment_number,
            'total_segments': total_segments,
            'origin_stop_id': origen['stop_id'],
            'origin_stop_name': origen['stop_name'],
            'destination_stop_id': destino['stop_id'],
            'destination_stop_name': destino['stop_name'],
            
            # Tiempos y delays (TARGET = delay_at_destination)
            'travel_time_minutes': time_diff,
            'delay_at_origin': origen['delay_minutes'],
            'delay_at_destination': destino['delay_minutes'], #target 
            'cumulative_delay_origin': origen['cumulative_delay'],
            'cumulative_delay_destination': destino['cumulative_delay'],
            
            # Features temporales basicas
            'hour': hour,
            'day_of_week': origen['dow'],
            'is_peak_hour': origen['is_peak_hour'],
            
            # Features temporales avanzadas
            'is_morning_rush': is_morning_rush,
            'is_lunch_rush': is_lunch_rush,
            'is_evening_rush': is_evening_rush,
            'minute_of_hour': minute_of_hour,
            'minute_of_day': minute,
            'is_night': is_night,
            
            # Features de posicion
            'position_ratio': position_ratio,
            'is_early_segment': is_early_segment,
            'is_late_segment': is_late_segment,
            
            # Features de contexto historico
            'delay_prev_1': delay_prev_1,
            'delay_prev_2': delay_prev_2,
            'delay_mean_prev': delay_mean_prev,
            'delay_std_prev': delay_std_prev,
            
            # Features de linea
            'is_line_3': is_line_3,
            'is_line_5': is_line_5,
            'is_express': is_express,
            
            # Features derivadas
            'delay_acceleration': delay_acceleration,
            
            # Features de interaccion
            'hour_x_position': hour_x_position,
            'rush_x_segment': rush_x_segment,

            # Tipos de servicio como feature
            'origin_service_type': get_stop_service_type(route_id, origen['stop_id']),
            'dest_service_type': get_stop_service_type(route_id, destino['stop_id']),

            # Features binarias
            'origin_is_part_time': get_stop_service_type(route_id, origen['stop_id']) == 'part_time',
            'dest_is_part_time': get_stop_service_type(route_id, destino['stop_id']) == 'part_time',
            'origin_is_rush_hour': get_stop_service_type(route_id, origen['stop_id']) == 'rush_hour_only',
            'dest_is_rush_hour': get_stop_service_type(route_id, destino['stop_id']) == 'rush_hour_only',
            
            # Metadata
            'timestamp': origen['timestamp']
        }
        
        viajes.append(viaje)

print(f"\nSegmentos creados: {len(viajes):,}")
print(f"Trenes omitidos (< 2 paradas): {skipped_too_short}")
print(f"Trenes omitidos (tiempos inconsistentes): {skipped_time_issues}")

# Crear DataFrame final

viajes_df = pd.DataFrame(viajes)
print(f"Shape: {viajes_df.shape}")
print(f"Columnas: {viajes_df.shape[1]}")

# Estadisticas del target (delay_at_destination)
print(f"\nEstadisticas de delay_at_destination:")
print(f"Media:    {viajes_df['delay_at_destination'].mean():.2f} min")
print(f"Mediana:  {viajes_df['delay_at_destination'].median():.2f} min")
print(f"Std:      {viajes_df['delay_at_destination'].std():.2f} min")
print(f"Min:      {viajes_df['delay_at_destination'].min():.2f} min")
print(f"Max:      {viajes_df['delay_at_destination'].max():.2f} min")
print(f"Q95:      {viajes_df['delay_at_destination'].quantile(0.95):.2f} min")
print(f"Q99:      {viajes_df['delay_at_destination'].quantile(0.99):.2f} min")

# Distribucion de delays
adelantos = (viajes_df['delay_at_destination'] < -0.5).sum()
puntuales = ((viajes_df['delay_at_destination'] >= -0.5) & 
             (viajes_df['delay_at_destination'] <= 0.5)).sum()
retrasos = (viajes_df['delay_at_destination'] > 0.5).sum()

print(f"\nDistribucion:")
print(f"Adelantos: {adelantos:,} ({adelantos/len(viajes_df)*100:.1f}%)")
print(f"Puntuales: {puntuales:,} ({puntuales/len(viajes_df)*100:.1f}%)")
print(f"Retrasos:  {retrasos:,} ({retrasos/len(viajes_df)*100:.1f}%)")

# Verificar NULLs (no deberia haber,pero por si acasi)
nulls = viajes_df.isnull().sum()
if nulls.sum() > 0:
    print(f"\nColumnas con NULLs:")
    for col, count in nulls[nulls > 0].items():
        print(f"  {col}: {count}")
else:
    print(f"\nNo hay valores NULL")


# Guardar como pickle (formato eficiente para pandas)
viajes_df.to_pickle('dataset_viajes_raw.pkl')
print(f"Guardado: dataset_viajes_raw.pkl ({viajes_df.shape[0]:,} registros)")


viajes_df.to_csv('dataset_viajes_raw.csv', index=False, sep=";")


# Guardar metadata
metadata = {
    'fecha_creacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_registros': len(viajes_df),
    'total_columnas': viajes_df.shape[1],
    'columnas': list(viajes_df.columns),
    'lineas': viajes_df['route_id'].unique().tolist(),
    'rango_delays': {
        'min': float(viajes_df['delay_at_destination'].min()),
        'max': float(viajes_df['delay_at_destination'].max()),
        'mean': float(viajes_df['delay_at_destination'].mean()),
        'median': float(viajes_df['delay_at_destination'].median())
    },
    'filtros_aplicados': [
        'NULL en campos criticos',
        'travel_time < 0.5 min (30 segundos)',
        'misma parada origen/destino',
        'tiempos no monotonicos'
    ]
}


# Resumen 

print(f"\nResumen:")
print(f"Total segmentos:  {len(viajes_df):,}")
print(f"Features:         {viajes_df.shape[1]}")
print(f"Lineas:           {len(viajes_df['route_id'].unique())}")
print(f"Trenes unicos:    {len(viajes_df['train_id'].unique())}")
print(f"Rango delays:     {viajes_df['delay_at_destination'].min():.1f} a {viajes_df['delay_at_destination'].max():.1f} min")
print(f"\nArchivos generados:")
print(f"dataset_viajes_raw.pkl")
print(f"dataset_viajes_raw.csv")
