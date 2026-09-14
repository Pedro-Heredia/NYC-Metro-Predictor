# Preparador del Dataset (pkl) para el EDA
#
# Version vectorizada (2026-07-07): la version anterior construia un dict de
# Python por cada segmento de viaje y los acumulaba en una lista antes de
# convertirla a DataFrame. Con >2M segmentos esto agotaba la RAM del servidor
# (7.2GB) y el proceso moria por OOM-kill (ver retrain.log, fallos 2026-06-28
# y 2026-07-05). Esta version usa operaciones vectorizadas de pandas sobre
# columnas completas, manteniendo exactamente la misma logica de negocio.
#
# Ademas se ha corregido una no-determinismo latente del script original:
# el orden de filas para desempatar timestamps/scheduled_time identicos
# (frecuente porque un mismo ciclo del collector registra varias paradas de
# un tren con el mismo timestamp) dependia del orden de entrega de Postgres,
# que no esta garantizado. Ahora se usa el id (PK) como desempate explicito,
# haciendo el dataset reproducible entre ejecuciones sobre los mismos datos.

import gc
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from service_types import _service_types_raw
warnings.filterwarnings('ignore')


def parse_gtfs_time(t):
    # GTFS permite horas >= 24 (ej: 24:05:00) para trenes que cruzan medianoche
    try:
        h, m, s = str(t).split(':')
        return int(h) * 60 + int(m) + int(s) / 60
    except:
        return np.nan


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
# id se usa unicamente como desempate deterministico de orden, no como feature.
# Sin ORDER BY: el orden final lo impone pandas mas abajo de todos modos, y
# pedirselo a Postgres obligaba a un sort completo de toda la tabla (lento y
# con un scan por indice de acceso aleatorio en tablas de varios millones de
# filas) para un resultado que se iba a reordenar igualmente.
query = """
    SELECT
        id,
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
        direction,
        timestamp
    FROM delays
    WHERE
        scheduled_time IS NOT NULL
        AND estimated_time IS NOT NULL
        AND delay_minutes IS NOT NULL
        AND cumulative_delay IS NOT NULL
"""

# Categorias unificadas calculadas de antemano: si cada lote se categoriza por
# separado, cada uno acaba con su propia tabla de categorias (subconjunto de
# valores vistos en ESE lote) y pd.concat tiene que reconciliarlas con
# union_categoricals, lo que internamente decodifica todo a texto otra vez y
# dispara el pico de RAM (se observo en pruebas: >6GB durante el concat).
# Con un CategoricalDtype fijo e identico en todos los lotes, concatenar es
# una simple concatenacion de arrays de codigos, sin reconciliacion.
CATEGORICAL_COLS = ('route_id', 'stop_id', 'stop_name', 'direction', 'train_id', 'scheduled_time')
cat_cur = conn.cursor()
category_dtypes = {}
for col in CATEGORICAL_COLS:
    cat_cur.execute(f"SELECT DISTINCT {col} FROM delays WHERE {col} IS NOT NULL")
    values = sorted(r[0] for r in cat_cur.fetchall())
    category_dtypes[col] = pd.CategoricalDtype(categories=values)
cat_cur.close()

# Cursor server-side + lectura por lotes: con un cursor normal, psycopg2/libpq
# vuelca TODO el resultado (varios millones de filas como tuplas de Python) en
# memoria del cliente antes de que pandas construya el DataFrame, duplicando
# el pico de RAM durante la lectura. Con un cursor con nombre, Postgres envia
# los datos en lotes bajo demanda y cada lote se compacta a dtypes eficientes
# (category/datetime) antes de acumularlo, evitando ese pico.
# Columnas en el mismo orden que el SELECT (evita depender de cur.description,
# que con cursores server-side no esta garantizado hasta el primer fetch).
columns = ['id', 'train_id', 'route_id', 'stop_id', 'stop_name', 'estimated_time',
           'scheduled_time', 'delay_minutes', 'cumulative_delay', 'hour', 'dow',
           'is_peak_hour', 'direction', 'timestamp']

CHUNK_SIZE = 300_000
cur = conn.cursor(name='prepare_dataset_cursor')
cur.itersize = CHUNK_SIZE
cur.execute(query)

chunks = []
total_rows = 0
while True:
    rows = cur.fetchmany(CHUNK_SIZE)
    if not rows:
        break
    chunk = pd.DataFrame(rows, columns=columns)
    for col in CATEGORICAL_COLS:
        chunk[col] = chunk[col].astype(category_dtypes[col])
    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
    try:
        chunk['estimated_time'] = pd.to_datetime(chunk['estimated_time'], format='%H:%M:%S', errors='coerce')
    except Exception:
        chunk['estimated_time'] = pd.to_datetime(chunk['estimated_time'], errors='coerce')
    chunks.append(chunk)
    total_rows += len(chunk)

cur.close()
conn.close()

df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"Datos extraidos, Nº de registros: {len(df)} ")

# Separar por fecha en hora NYC para que el mismo trip_id de dias distintos
# no se mezcle en el mismo grupo (un trip_id se repite cada dia laborable).
# normalize() en vez de .dt.date: mismo valor logico pero como datetime64
# (8 bytes) en vez de objetos date de Python (~50 bytes cada uno).
df['fecha'] = df['timestamp'].dt.normalize()

# Separar runs: el mismo train_id puede reutilizarse en el mismo dia calendario.
# Un gap >1h entre registros consecutivos del mismo tren indica runs distintos
# (ej: tren de medianoche visible a las 23:30 del dia N y a las 00:05 del dia N+1,
# ambos con fecha=N pero son viajes distintos).
df = df.sort_values(['train_id', 'fecha', 'timestamp', 'id'])
run_group_cols = ['train_id', 'fecha']
ts_gap = df.groupby(run_group_cols, observed=True)['timestamp'].diff().dt.total_seconds().fillna(0)
df['_run_id'] = (ts_gap > 3600).astype(int)
df['_run_id'] = df.groupby(run_group_cols, observed=True)['_run_id'].cumsum()
group_cols = ['train_id', 'fecha', '_run_id']
n_runs = df.groupby(group_cols, observed=True).ngroups
print(f"Runs identificados: {n_runs:,}")
del ts_gap
gc.collect()

# --- Construccion vectorizada de segmentos de viaje ---
# (equivalente al bucle fila a fila original, verificado por comparacion
# exacta sobre datos reales antes de desplegar)

# 1) Deduplicar: misma parada puede aparecer en multiples ciclos consecutivos.
#    Quedarse con la observacion mas reciente (delay mas actualizado) por run.
df = df.sort_values(group_cols + ['timestamp', 'id'])
df = df.drop_duplicates(subset=group_cols + ['stop_id'], keep='last')
gc.collect()

# 2) Convertir scheduled_time a minutos y ordenar por ruta (orden real de paradas)
# .astype(float): Series.apply() sobre una columna categorica aplica la
# funcion solo a las categorias unicas (optimizacion de pandas) y devuelve
# el resultado TODAVIA como Categorical (con las categorias ya convertidas a
# minutos) en vez de un float64 plano; sin este cast, las operaciones
# aritmeticas posteriores (travel_time_minutes) fallan.
df['sched_minutes'] = df['scheduled_time'].apply(parse_gtfs_time).astype(float)
df = df.sort_values(group_cols + ['sched_minutes', 'id'])
df.drop(columns=['scheduled_time', 'id'], inplace=True)
gc.collect()

# 3) Filtrar runs con menos de 2 paradas (tras el dedup)
sizes = df.groupby(group_cols, observed=True).size()
skipped_too_short = int((sizes < 2).sum())
df = df[df.groupby(group_cols, observed=True)['stop_id'].transform('size') >= 2]

# 4) Eliminar filas con estimated_time no parseable (NaT)

df = df.dropna(subset=['estimated_time'])
sizes_after_nat = df.groupby(group_cols, observed=True).size()
skipped_too_short += int((sizes_after_nat < 2).sum())
df = df[df.groupby(group_cols, observed=True)['stop_id'].transform('size') >= 2]

skipped_time_issues = 0  # dead branch tambien en el script original (nunca se incrementaba)

# 5) Features de contexto historico (delay_prev_1/2, medias y std expandiendo)
gb_delay = df.groupby(group_cols, observed=True)['delay_minutes']
df['delay_prev_1'] = gb_delay.shift(1)
df['delay_prev_2'] = gb_delay.shift(2)

cumsum = gb_delay.cumsum()
# Cuadrado vectorizado + cumsum nativo de groupby, en vez de un lambda por
# grupo (198k+ grupos): mismo resultado, mucho mas rapido a esta escala.
df['_delay_sq'] = df['delay_minutes'] ** 2
cumsq = df.groupby(group_cols, observed=True)['_delay_sq'].cumsum()
n = df.groupby(group_cols, observed=True).cumcount() + 1
exp_mean = cumsum / n
exp_var = (cumsq - (cumsum ** 2) / n) / (n - 1)
exp_std = np.sqrt(exp_var.where(n > 1))
df.drop(columns=['_delay_sq'], inplace=True)

df['_exp_mean'] = exp_mean
df['_exp_std'] = exp_std
df['delay_mean_prev'] = df.groupby(group_cols, observed=True)['_exp_mean'].shift(1)
df['delay_std_prev'] = df.groupby(group_cols, observed=True)['_exp_std'].shift(1)
df.drop(columns=['_exp_mean', '_exp_std'], inplace=True)

# Replica fiel del script original: "origen['delay_prev_1'] is not None" nunca
# es False para NaN (NaN is not None en Python), asi que prev_1/prev_2 se
# dejan tal cual (con NaN incluido si es el inicio del run). mean/std si
# comprueban pd.isna() y por tanto SI se rellenan.
df['delay_mean_prev'] = df['delay_mean_prev'].fillna(0)
df['delay_std_prev'] = df['delay_std_prev'].fillna(0.5)

# 6) Posicion dentro del run
df['segment_number'] = df.groupby(group_cols, observed=True).cumcount() + 1
df['total_segments'] = df.groupby(group_cols, observed=True)['stop_id'].transform('size') - 1
df['position_ratio'] = np.where(df['total_segments'] > 0,
                                 df['segment_number'] / df['total_segments'], 0.5)
df['is_early_segment'] = (df['position_ratio'] < 0.33).astype(int)
df['is_late_segment'] = (df['position_ratio'] > 0.67).astype(int)

# 7) Features temporales
df['minute_of_day'] = (df['estimated_time'].dt.hour * 60 + df['estimated_time'].dt.minute) % 1440
df['minute_of_hour'] = df['estimated_time'].dt.minute
df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)

# 8) Features de linea (las lineas 3, 5, 7 son express oficialmente segun MTA;
#    ademas comprobado por anteriores EDAs que las lineas 3 y 5 son las mas
#    problematicas, son las que mas se retrasan)
df['is_line_3'] = (df['route_id'] == '3').astype(int)
df['is_line_5'] = (df['route_id'] == '5').astype(int)
df['is_express'] = df['route_id'].isin(['3', '5', '7']).astype(int)

# 9) Features de interaccion
df['hour_x_position'] = df['hour'] * df['position_ratio']
df['rush_x_segment'] = (df['is_morning_rush'] + df['is_evening_rush']) * df['segment_number']

# 10) Destino: desplazar una posicion dentro de cada run
gb = df.groupby(group_cols, observed=True)
df['destination_stop_id'] = gb['stop_id'].shift(-1)
df['destination_stop_name'] = gb['stop_name'].shift(-1)
df['_dest_sched_minutes'] = gb['sched_minutes'].shift(-1)
df['delay_at_destination'] = gb['delay_minutes'].shift(-1)
df['cumulative_delay_destination'] = gb['cumulative_delay'].shift(-1)
df['travel_time_minutes'] = df['_dest_sched_minutes'] - df['sched_minutes']

# 11) Tipos de servicio (lookup vectorizado; el original usa siempre el
#     route_id del origen tanto para origin_service_type como dest_service_type)
service_key_origin = df['route_id'].astype(str) + '|' + df['stop_id'].astype(str)
df['origin_service_type'] = service_key_origin.map(_service_types_raw).fillna('full_time')

service_key_dest = df['route_id'].astype(str) + '|' + df['destination_stop_id'].astype(str)
df['dest_service_type'] = service_key_dest.map(_service_types_raw).fillna('full_time')

df['origin_is_part_time'] = df['origin_service_type'] == 'part_time'
df['dest_is_part_time'] = df['dest_service_type'] == 'part_time'
df['origin_is_rush_hour'] = df['origin_service_type'] == 'rush_hour_only'
df['dest_is_rush_hour'] = df['dest_service_type'] == 'rush_hour_only'

# 12) Features temporales/direccion adicionales
df['month'] = pd.to_datetime(df['fecha']).dt.month
df['is_weekend'] = (df['dow'] >= 5).astype(int)
df['direction_north'] = (df['direction'] == 'NORTH').astype(int)

# 13) Filtrar filas que no forman un segmento valido:
#     - ultima parada de cada run (destino NaN tras el shift)
#     - tiempo de viaje imposible (<30s) o desconocido
#     - misma parada origen/destino
valid = (
    df['destination_stop_id'].notna()
    & df['travel_time_minutes'].notna()
    & (df['travel_time_minutes'] >= 0.5)
    & (df['stop_id'].astype(str) != df['destination_stop_id'].astype(str))
)
viajes_df = df[valid].copy()
del df
gc.collect()

# Crear DataFrame final
viajes_df = viajes_df.rename(columns={
    'stop_id': 'origin_stop_id',
    'stop_name': 'origin_stop_name',
    'delay_minutes': 'delay_at_origin',
    'cumulative_delay': 'cumulative_delay_origin',
    'dow': 'day_of_week',
})

FINAL_COLUMNS = [
    'train_id', 'route_id', 'segment_number', 'total_segments',
    'origin_stop_id', 'origin_stop_name', 'destination_stop_id', 'destination_stop_name',
    'travel_time_minutes', 'delay_at_origin', 'delay_at_destination',
    'cumulative_delay_origin', 'cumulative_delay_destination',
    'hour', 'day_of_week', 'is_peak_hour',
    'is_morning_rush', 'is_evening_rush', 'minute_of_hour', 'minute_of_day', 'is_night',
    'position_ratio', 'is_early_segment', 'is_late_segment',
    'delay_prev_1', 'delay_prev_2', 'delay_mean_prev', 'delay_std_prev',
    'is_line_3', 'is_line_5', 'is_express',
    'hour_x_position', 'rush_x_segment',
    'origin_service_type', 'dest_service_type',
    'origin_is_part_time', 'dest_is_part_time', 'origin_is_rush_hour', 'dest_is_rush_hour',
    'month', 'is_weekend', 'direction_north',
    'timestamp',
]
viajes_df = viajes_df[FINAL_COLUMNS].reset_index(drop=True)

for col in ('train_id', 'route_id', 'origin_stop_id', 'origin_stop_name',
            'destination_stop_id', 'destination_stop_name'):
    viajes_df[col] = viajes_df[col].astype(str)

print(f"Shape: {viajes_df.shape}")
print(f"Columnas: {viajes_df.shape[1]}")

print(f"\nSegmentos creados: {len(viajes_df):,}")
print(f"Trenes omitidos (< 2 paradas): {skipped_too_short}")
print(f"Trenes omitidos (tiempos inconsistentes): {skipped_time_issues}")

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
