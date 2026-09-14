#Recolector multi-lineas MTA

import sys
import re
import random
import requests
import pandas as pd
from google.transit import gtfs_realtime_pb2
from datetime import datetime, timedelta
import psycopg2
import time
import pytz
import pickle
from pathlib import Path


FEEDS = {
    '1-2-3-4-5-6-7': 'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs',
    'A-C-E':         'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace',
    'B-D-F-M':       'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-bdfm',
    'G':             'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-g',
    'J-Z':           'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-jz',
    'N-Q-R-W':       'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-nqrw',
    'L':             'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-l',
}

TARGET_ROUTES = {'1','2','3','4','5','6','7','A','C','E','B','D','F','M','G','J','Z','N','Q','R','W','L'}
TRAINS_PER_LINE = 6          # muestra aleatoria (shuffle previo), ligeramente aumentada
MAX_FUTURE_MINUTES = 30      # descartar paradas predichas a mas de 30 min vista
COLLECTION_INTERVAL = 300
DEBUG_MODE = False
GTFS_DIR = Path(".")

# GTFS Supplemented: update_gtfs_supplemented.py (cron horario) descarga el feed
# con desviaciones/obras de los proximos 7 dias, construye los indices con la
# misma load_gtfs_static() y los deja en SUPPLEMENTED_INDICES_PATH via os.replace
# (atomico). El collector solo LEE ese pickle si ha cambiado; nunca reconstruye
# indices el mismo, asi que nunca bloquea un ciclo de 5 min.
SUPPLEMENTED_INDICES_PATH = Path("/home/pedro/NYC/gtfs_supplemented_indices.pkl")

DB_CONFIG = {
    "host": "localhost",
    "dbname": "delays2",
    "user": "postgres",
    "password": "tfg"
}

# Claves (train_id, stop_id, scheduled_time) del ciclo anterior para deduplicacion cross-ciclo
_prev_cycle_seen = set()


def get_ny_time():
    tz_ny = pytz.timezone("America/New_York")
    return datetime.now(tz_ny)


def load_gtfs_static(gtfs_dir=None):
    # gtfs_dir permite reutilizar esta misma logica de parseo/indexado tanto
    # para el GTFS regular (por defecto GTFS_DIR) como para el Supplemented
    # (usado por update_gtfs_supplemented.py), sin duplicar codigo.
    gtfs_dir = gtfs_dir or GTFS_DIR
    try:
        stops_df = pd.read_csv(gtfs_dir / "stops.txt")
        stop_times_df = pd.read_csv(gtfs_dir / "stop_times.txt")
        trips_df = pd.read_csv(gtfs_dir / "trips.txt")
        calendar_df = pd.read_csv(gtfs_dir / "calendar.txt")
        try:
            calendar_dates_df = pd.read_csv(gtfs_dir / "calendar_dates.txt")
        except Exception:
            calendar_dates_df = pd.DataFrame()

        print("Construyendo indice stop_times...")
        # {trip_id: {stop_id: arrival_time}} - lookup O(1) en vez de scan O(n)
        stop_times_by_trip = {}
        for trip_id, group in stop_times_df.groupby('trip_id'):
            stop_times_by_trip[trip_id] = dict(zip(group['stop_id'], group['arrival_time']))
        print(f"Indice principal: {len(stop_times_by_trip)} trips indexados")

        # Indices para match exacto por trip_id string (evita fuzzy matching cuando es posible)
        # El trip_id estatico tiene formato: {prefix}_{numero}_{ruta}..{dir}{terminus}
        # El trip_id RT tiene formato:       {numero}_{ruta}..{dir}[{terminus}]
        # trip_suffix_index: {numero_ruta_dir_terminus -> static_trip_id}  (match exacto)
        # trip_prefix_index: {numero_ruta_dir -> static_trip_id}           (RT sin terminus)
        print("Construyendo indices de matching por trip_id...")
        trip_suffix_index = {}
        trip_prefix_index = {}
        trip_service_dict = trips_df.set_index('trip_id')['service_id'].to_dict()
        trip_route_dict   = trips_df.set_index('trip_id')['route_id'].to_dict()

        for tid in trips_df['trip_id']:
            m = re.search(r'(\d+_.*)', tid)
            if not m:
                continue
            suffix = m.group(1)
            trip_suffix_index[suffix] = tid
            # Base sin terminus: '046950_2..S06R' -> '046950_2..S'
            base_m = re.match(r'(\d+_\w+\.\.[NS])', suffix)
            if base_m:
                base = base_m.group(1)
                if base not in trip_prefix_index:
                    trip_prefix_index[base] = tid
        print(f"Indices de trip_id: {len(trip_suffix_index)} exactos, {len(trip_prefix_index)} por prefijo")

        # {route_id: {base_stop_id: [time_str, ...]}} - fallback cuando el trip matched
        # no contiene una parada (mismatch express/local u otras variaciones de servicio)
        print("Construyendo indice de fallback por ruta...")
        trip_to_route = trips_df.set_index('trip_id')['route_id'].to_dict()
        route_stop_index = {}
        for trip_id, stops_dict in stop_times_by_trip.items():
            route_id = trip_to_route.get(trip_id)
            if not route_id or route_id not in TARGET_ROUTES:
                continue
            if route_id not in route_stop_index:
                route_stop_index[route_id] = {}
            for stop_id, time_str in stops_dict.items():
                base_id = stop_id.rstrip('NS')
                if base_id not in route_stop_index[route_id]:
                    route_stop_index[route_id][base_id] = []
                route_stop_index[route_id][base_id].append(time_str)
        print(f"Indice fallback: {sum(len(v) for v in route_stop_index.values())} paradas unicas en ruta")

        return (stops_df, stop_times_by_trip, route_stop_index,
                trip_suffix_index, trip_prefix_index, trip_service_dict, trip_route_dict,
                trips_df, calendar_df, calendar_dates_df)

    except FileNotFoundError as e:
        print(f"Error cargando GTFS: {e}")
        return None, None, None, None, None


def time_to_minutes(time_str):
    if pd.isna(time_str) or time_str == '':
        return None
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])


def get_stop_name(stop_id, stops_df):
    base_stop_id = stop_id.rstrip('NS')
    stop_row = stops_df[stops_df['stop_id'] == stop_id]
    if not stop_row.empty:
        return stop_row.iloc[0]['stop_name']
    stop_row = stops_df[stops_df['stop_id'] == base_stop_id]
    if not stop_row.empty:
        return stop_row.iloc[0]['stop_name']
    return stop_id


def calculate_delay_minutes(scheduled_time_str, estimated_datetime):
    if not scheduled_time_str:
        return None
    scheduled_minutes = time_to_minutes(scheduled_time_str)
    estimated_minutes = estimated_datetime.hour * 60 + estimated_datetime.minute
    if scheduled_minutes is None:
        return None
    delay = estimated_minutes - scheduled_minutes
    if abs(delay) > 720:
        delay_options = [
            delay,
            (estimated_minutes + 1440) - scheduled_minutes,
            estimated_minutes - (scheduled_minutes + 1440),
            (estimated_minutes - 1440) - scheduled_minutes
        ]
        delay = min(delay_options, key=abs)
    return delay


def get_current_service_ids(calendar_df, calendar_dates_df):
    # Usar hora NYC: el dia de servicio de la MTA es el de Nueva York, no el del servidor
    ny_tz = pytz.timezone("America/New_York")
    today_ny = datetime.now(ny_tz)
    weekday = today_ny.weekday()
    today_str = today_ny.strftime('%Y%m%d')
    weekday_map = {
        0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday',
        4: 'friday', 5: 'saturday', 6: 'sunday'
    }
    current_day = weekday_map[weekday]
    active_services = set()

    for _, service in calendar_df.iterrows():
        if service[current_day] == 1:
            active_services.add(service['service_id'])

    if not calendar_dates_df.empty:
        today_exceptions = calendar_dates_df[calendar_dates_df['date'] == int(today_str)]
        for _, exception in today_exceptions.iterrows():
            if exception['exception_type'] == 1:
                active_services.add(exception['service_id'])
            elif exception['exception_type'] == 2:
                active_services.discard(exception['service_id'])

    return list(active_services)


def match_trip_by_id(rt_trip_id, route_id, active_services,
                     trip_suffix_index, trip_prefix_index,
                     trip_service_dict, trip_route_dict):
    # Intento 1: match exacto — RT '046950_2..S06R' == sufijo de estatico
    static_id = trip_suffix_index.get(rt_trip_id)
    if (static_id
            and trip_route_dict.get(static_id) == route_id
            and trip_service_dict.get(static_id) in active_services):
        return static_id

    # Intento 2: RT sin terminus '047750_2..S' -> estatico tiene '047750_2..S06R'
    base_m = re.match(r'(\d+_\w+\.\.[NS])', rt_trip_id)
    if base_m:
        static_id = trip_prefix_index.get(base_m.group(1))
        if (static_id
                and trip_route_dict.get(static_id) == route_id
                and trip_service_dict.get(static_id) in active_services):
            return static_id

    return None  # tren dinamico: usar fuzzy matching por paradas


def find_scheduled_trip(future_stops, route_id, direction_hint, trips_df, stop_times_by_trip, active_services):
    # Filtrar trips activos de esta ruta
    route_trips = trips_df[
        (trips_df['route_id'] == route_id) &
        (trips_df['service_id'].isin(active_services))
    ]
    if direction_hint:
        direction_id = 1 if 'S' in direction_hint else 0
        route_trips = route_trips[route_trips['direction_id'] == direction_id]

    if route_trips.empty:
        route_trips = trips_df[trips_df['route_id'] == route_id]
        if direction_hint:
            direction_id = 1 if 'S' in direction_hint else 0
            route_trips = route_trips[route_trips['direction_id'] == direction_id]

    trip_ids = route_trips['trip_id'].tolist()
    if not trip_ids:
        return None

    stops_to_check = min(5, len(future_stops))
    best_trip_id = None
    best_score = float('inf')

    for trip_id in trip_ids:
        trip_stops = stop_times_by_trip.get(trip_id)
        if not trip_stops:
            continue

        # Validar horario con primera parada que coincida
        first_match_sched = None
        first_match_real = None
        for stop_data in future_stops[:stops_to_check]:
            stop_id = stop_data['stop_id']
            sched = trip_stops.get(stop_id) or trip_stops.get(stop_id.rstrip('NS'))
            if sched:
                first_match_sched = sched
                first_match_real = stop_data['arrival_time']
                break

        if first_match_sched:
            sched_min = time_to_minutes(first_match_sched)
            real_min = first_match_real.hour * 60 + first_match_real.minute
            if sched_min and sched_min < 360 and real_min > 1200:
                sched_min += 1440
            if sched_min:
                diff = min(
                    abs(real_min - sched_min),
                    abs((real_min + 1440) - sched_min),
                    abs(real_min - (sched_min + 1440))
                )
                if diff > 30:
                    continue

        total_diff = 0
        matched_stops = 0
        for stop_data in future_stops[:stops_to_check]:
            stop_id = stop_data['stop_id']
            real_time = stop_data['arrival_time']
            sched_str = trip_stops.get(stop_id) or trip_stops.get(stop_id.rstrip('NS'))
            if not sched_str:
                continue
            sched_min = time_to_minutes(sched_str)
            real_min = real_time.hour * 60 + real_time.minute
            if sched_min and sched_min < 360 and real_min > 1200:
                sched_min += 1440
            if sched_min is not None:
                diff = min(
                    abs(real_min - sched_min),
                    abs((real_min + 1440) - sched_min),
                    abs(real_min - (sched_min + 1440))
                )
                total_diff += diff
                matched_stops += 1

        if matched_stops > 0:
            avg_diff = total_diff / matched_stops
            coverage_penalty = (stops_to_check - matched_stops) * 30
            final_score = avg_diff + coverage_penalty
            if final_score < best_score:
                best_score = final_score
                best_trip_id = trip_id
                if final_score < 5:
                    break

    return best_trip_id


def get_scheduled_times(trip_id, stop_ids, stop_times_by_trip):
    scheduled_times = {}
    if not trip_id:
        return scheduled_times
    trip_stops = stop_times_by_trip.get(trip_id, {})
    for stop_id in stop_ids:
        sched = trip_stops.get(stop_id) or trip_stops.get(stop_id.rstrip('NS'))
        if sched:
            scheduled_times[stop_id] = sched
    return scheduled_times


def calculate_cumulative_delay(future_stops, scheduled_times):
    cumulative_delays = {}
    for i, stop_data in enumerate(future_stops):
        stop_id = stop_data['stop_id']
        arrival_time = stop_data['arrival_time']
        scheduled_time_str = scheduled_times.get(stop_id)
        if scheduled_time_str:
            delay_minutes = calculate_delay_minutes(scheduled_time_str, arrival_time)
            if delay_minutes is not None:
                if i == 0:
                    cumulative_delays[stop_id] = delay_minutes
                else:
                    prev = list(cumulative_delays.values())[-1] if cumulative_delays else 0
                    cumulative_delays[stop_id] = max(prev, delay_minutes) if prev is not None else delay_minutes
            else:
                cumulative_delays[stop_id] = None
        else:
            cumulative_delays[stop_id] = None
    return cumulative_delays


def normalize_time_format(time_str):
    if not time_str or time_str == "N/A":
        return time_str
    parts = time_str.split(':')
    if len(parts) >= 2:
        hours = int(parts[0])
        if hours >= 24:
            hours -= 24
        return f"{hours:02d}:{parts[1]}:{parts[2] if len(parts) > 2 else '00'}"
    return time_str


def is_peak_hour(ny_time):
    hour = ny_time.hour
    minute = ny_time.minute
    current_minutes = hour * 60 + minute
    # Hora punta NYC: manana 7-9h, tarde 17-19h
    peak_ranges = [(7 * 60, 9 * 60), (17 * 60, 19 * 60)]
    return any(start <= current_minutes <= end for start, end in peak_ranges)


def collect_data(stops_df, stop_times_by_trip, route_stop_index,
                 trip_suffix_index, trip_prefix_index, trip_service_dict, trip_route_dict,
                 trips_df, calendar_df, calendar_dates_df, conn=None):
    global _prev_cycle_seen
    current_cycle_seen = set()

    ny_time = get_ny_time()
    current_time = datetime.now()
    active_services = get_current_service_ids(calendar_df, calendar_dates_df)

    print(f"Hora NYC: {ny_time.strftime('%H:%M:%S')} | Servicios activos: {len(active_services)}")

    trains_by_route = {}
    all_data = []
    discarded = 0

    for feed_name, feed_url in FEEDS.items():
        try:
            response = requests.get(feed_url, timeout=30)
            if response.status_code != 200:
                print(f"  Feed {feed_name}: Error HTTP {response.status_code}")
                continue

            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(response.content)

            # Mezclar entidades aleatoriamente para muestreo sin sesgo posicional
            entities = list(feed.entity)
            random.shuffle(entities)

            for entity in entities:
                if not entity.HasField('trip_update'):
                    continue

                trip = entity.trip_update
                trip_id = trip.trip.trip_id
                route_id = trip.trip.route_id

                if route_id not in TARGET_ROUTES:
                    continue
                if route_id not in trains_by_route:
                    trains_by_route[route_id] = []
                if len(trains_by_route[route_id]) >= TRAINS_PER_LINE:
                    continue

                direction = "DESCONOCIDA"
                direction_hint = None
                if '..' in trip_id:
                    parts = trip_id.split('..')
                    if len(parts) > 1 and parts[1]:
                        direction_hint = parts[1][0]
                        if direction_hint == 'S':
                            direction = "SOUTH"
                        elif direction_hint == 'N':
                            direction = "NORTH"

                future_stops = []
                for stop_update in trip.stop_time_update:
                    if stop_update.HasField('arrival'):
                        arrival_time = datetime.fromtimestamp(stop_update.arrival.time)
                        minutes_until = (arrival_time - current_time).total_seconds() / 60
                        # Solo paradas proximas: descartar predicciones lejanas e inciertas
                        if minutes_until >= -1 and minutes_until <= MAX_FUTURE_MINUTES:
                            future_stops.append({
                                'stop_id': stop_update.stop_id,
                                'arrival_time': arrival_time,
                                'minutes_until': minutes_until
                            })

                if not future_stops:
                    continue

                future_stops.sort(key=lambda x: x['minutes_until'])

                # Match por trip_id string (exacto o por prefijo): cubre ~83% de trenes
                best_trip = match_trip_by_id(
                    trip_id, route_id, active_services,
                    trip_suffix_index, trip_prefix_index,
                    trip_service_dict, trip_route_dict
                )
                # Fallback fuzzy por patrones de paradas: solo para trenes dinamicos (~17%)
                if best_trip is None:
                    best_trip = find_scheduled_trip(
                        future_stops, route_id, direction_hint,
                        trips_df, stop_times_by_trip, active_services
                    )

                if not best_trip:
                    discarded += 1
                    continue

                stop_ids = [s['stop_id'] for s in future_stops]
                scheduled_times = get_scheduled_times(best_trip, stop_ids, stop_times_by_trip)

                # Fallback: paradas no encontradas en el trip matched se buscan en todos
                # los trips de la misma ruta (cubre mismatches express/local y diversiones)
                for stop_data in future_stops:
                    stop_id = stop_data['stop_id']
                    if stop_id in scheduled_times:
                        continue
                    base_id = stop_id.rstrip('NS')
                    candidates = route_stop_index.get(route_id, {}).get(base_id, [])
                    if not candidates:
                        continue
                    real_min = stop_data['arrival_time'].hour * 60 + stop_data['arrival_time'].minute
                    best_time = None
                    best_diff = 31  # solo aceptar si esta dentro de 30 min
                    for t in candidates:
                        mins = time_to_minutes(t)
                        if mins is None:
                            continue
                        diff = min(
                            abs(real_min - mins),
                            abs((real_min + 1440) - mins),
                            abs(real_min - (mins + 1440))
                        )
                        if diff < best_diff:
                            best_diff = diff
                            best_time = t
                    if best_time:
                        scheduled_times[stop_id] = best_time

                valid_scheduled = sum(1 for v in scheduled_times.values() if v is not None)
                if not future_stops or valid_scheduled / len(future_stops) < 0.6:
                    discarded += 1
                    continue

                # Validar primeras paradas con delay calculable (hasta 5 o las que haya)
                ok = True
                valid_first = 0
                n_to_check = min(5, len(future_stops))
                for stop_data in future_stops[:n_to_check]:
                    stop_id = stop_data['stop_id']
                    sched_str = scheduled_times.get(stop_id)
                    if not sched_str:
                        ok = False
                        break
                    delay = calculate_delay_minutes(sched_str, stop_data['arrival_time'])
                    if delay is None:
                        ok = False
                        break
                    valid_first += 1

                if not ok or valid_first < n_to_check:
                    discarded += 1
                    continue

                # Verificar consistencia de delays: alta desviacion tipica indica
                # mal matching (ej. trip local asignado a tren express o viceversa)
                all_delays = [
                    calculate_delay_minutes(scheduled_times[s['stop_id']], s['arrival_time'])
                    for s in future_stops if s['stop_id'] in scheduled_times
                ]
                all_delays = [d for d in all_delays if d is not None]
                if len(all_delays) >= 3:
                    mean_d = sum(all_delays) / len(all_delays)
                    std_d = (sum((d - mean_d) ** 2 for d in all_delays) / len(all_delays)) ** 0.5
                    if std_d > 8:
                        discarded += 1
                        continue

                cumulative_delays = calculate_cumulative_delay(future_stops, scheduled_times)
                trains_by_route[route_id].append(trip_id)

                # dow y hour siempre en hora NYC (el modelo aprende patrones de NYC)
                dow = ny_time.weekday()
                hour = ny_time.hour
                is_peak = is_peak_hour(ny_time)

                for stop_data in future_stops:
                    stop_id = stop_data['stop_id']
                    arrival_time = stop_data['arrival_time']
                    sched_str = scheduled_times.get(stop_id)

                    # Si despues del fallback sigue sin horario, es una parada
                    # genuinamente fuera de cualquier trip conocido: descartar
                    if not sched_str:
                        continue

                    delay_raw = calculate_delay_minutes(sched_str, arrival_time)
                    cum_delay = cumulative_delays.get(stop_id)

                    if delay_raw is None or cum_delay is None:
                        continue

                    # Deduplicacion cross-ciclo: mismo tren en misma parada con mismo
                    # horario programado es la misma observacion que ya guardamos antes
                    dedup_key = (trip_id, stop_id, sched_str)
                    if dedup_key in _prev_cycle_seen:
                        continue
                    current_cycle_seen.add(dedup_key)

                    stop_name = get_stop_name(stop_id, stops_df)

                    all_data.append({
                        'timestamp': ny_time.strftime("%Y-%m-%d %H:%M:%S"),
                        'train_id': trip_id,
                        'route_id': route_id,
                        'direction': direction,
                        'stop_id': stop_id,
                        'stop_name': stop_name,
                        'scheduled_time': sched_str,
                        'estimated_time': arrival_time.strftime("%H:%M:%S"),
                        'delay_minutes': round(delay_raw, 1),
                        'dow': dow,
                        'hour': hour,
                        'is_peak_hour': is_peak,
                        'cumulative_delay': round(cum_delay, 1)
                    })

        except Exception as e:
            print(f"  Feed {feed_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    _prev_cycle_seen = current_cycle_seen

    route_summary = {r: len(trains) for r, trains in trains_by_route.items()}
    total_trains = sum(route_summary.values())
    print(f"Trenes validos: {total_trains} | Descartados: {discarded} | Registros: {len(all_data)}")
    print(f"Lineas: {dict(sorted(route_summary.items()))}")

    if not DEBUG_MODE and conn is not None:
        try:
            cur = conn.cursor()
            values = [
                (
                    row['timestamp'], row['train_id'], row['route_id'], row['direction'],
                    row['stop_id'], row['stop_name'], row['scheduled_time'], row['estimated_time'],
                    row['delay_minutes'], row['dow'], row['hour'], row['is_peak_hour'],
                    row['cumulative_delay']
                )
                for row in all_data
            ]
            cur.executemany("""
                INSERT INTO delays (
                    timestamp, train_id, route_id, direction,
                    stop_id, stop_name, scheduled_time, estimated_time,
                    delay_minutes, dow, hour, is_peak_hour, cumulative_delay
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, values)
            conn.commit()
            cur.close()
            print(f"Guardados en BD: {len(all_data)} registros")
        except Exception as e:
            print(f"Error BD: {e}")
            try:
                conn.rollback()
            except Exception:
                pass

    return len(all_data)


def try_reload_supplemented_indices(last_mtime):
    # Lee el pickle de indices Supplemented si ha cambiado desde la ultima vez.
    # Nunca reconstruye nada aqui (eso lo hace update_gtfs_supplemented.py aparte);
    # solo deserializa un resultado ya calculado, por lo que es rapido (no bloquea
    # el ciclo de recoleccion). Cualquier fallo (fichero ausente, pickle corrupto,
    # etc.) se registra y se ignora: el collector sigue con los indices que ya tenia.
    if not SUPPLEMENTED_INDICES_PATH.exists():
        return None, last_mtime

    try:
        mtime = SUPPLEMENTED_INDICES_PATH.stat().st_mtime
    except Exception as e:
        print(f"GTFS Supplemented: no se pudo comprobar el fichero de indices ({e}).")
        return None, last_mtime

    if mtime == last_mtime:
        return None, last_mtime

    try:
        with open(SUPPLEMENTED_INDICES_PATH, 'rb') as f:
            indices = pickle.load(f)
        print(f"GTFS Supplemented: indices actualizados (mtime {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')})")
        return indices, mtime
    except Exception as e:
        print(f"GTFS Supplemented: fallo al recargar indices ({e}). Se mantienen los indices actuales.")
        return None, mtime


def main():
    print(f"NYC Collector iniciado")
    print(f"Trenes por linea: {TRAINS_PER_LINE} (muestreo aleatorio) | Intervalo: {COLLECTION_INTERVAL}s | Max futuro: {MAX_FUTURE_MINUTES}min")

    (stops_df, stop_times_by_trip, route_stop_index,
     trip_suffix_index, trip_prefix_index, trip_service_dict, trip_route_dict,
     trips_df, calendar_df, calendar_dates_df) = load_gtfs_static()
    if stops_df is None:
        print("Error cargando GTFS, saliendo")
        return

    # Si ya existe un pickle Supplemented de una ejecucion anterior del cron,
    # usarlo desde el arranque en vez de esperar hasta una hora a la siguiente
    # actualizacion. Si no existe todavia (primera vez), se sigue con el GTFS
    # regular hasta que el cron horario genere el primero.
    supplemented_mtime = None
    reloaded, supplemented_mtime = try_reload_supplemented_indices(supplemented_mtime)
    if reloaded is not None:
        (stops_df, stop_times_by_trip, route_stop_index,
         trip_suffix_index, trip_prefix_index, trip_service_dict, trip_route_dict,
         trips_df, calendar_df, calendar_dates_df) = reloaded
        print("GTFS Supplemented: indices cargados desde el arranque")

    # Reintentos con espera: un restart de postgres (p.ej. por una actualizacion
    # automatica del sistema) puede tardar unos segundos en volver a aceptar
    # conexiones. Sin esto, un fallo transitorio justo al arrancar mataba el
    # servicio entero SIN que systemd lo reintentara (ver mas abajo el porque).
    conn = None
    for attempt in range(5):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            print(f"BD conectada: {DB_CONFIG['dbname']}")
            break
        except Exception as e:
            print(f"Error conectando BD (intento {attempt + 1}/5): {e}")
            if attempt < 4:
                time.sleep(10)

    if conn is None:
        # sys.exit(1) en vez de return: un "return" aqui termina main() y el
        # proceso sale con codigo 0 (exito), y Restart=on-failure de systemd
        # NO reinicia servicios que terminan con exito. Con exit(1) systemd
        # si lo reintenta cada RestartSec (60s) hasta que la BD vuelva,
        # en vez de dejar el collector parado indefinidamente sin avisar.
        print("No se pudo conectar a la BD tras 5 intentos. Saliendo con error para que systemd reinicie el servicio.")
        sys.exit(1)

    iteration = 0
    while True:
        try:
            iteration += 1
            print(f"\n--- Iteracion #{iteration} ---")
            t_start = time.time()

            if conn is None or conn.closed != 0:
                try:
                    conn = psycopg2.connect(**DB_CONFIG)
                    print("Reconectado a BD")
                except Exception as e_conn:
                    print(f"Error reconectando BD: {e_conn}")
                    conn = None

            reloaded, supplemented_mtime = try_reload_supplemented_indices(supplemented_mtime)
            if reloaded is not None:
                (stops_df, stop_times_by_trip, route_stop_index,
                 trip_suffix_index, trip_prefix_index, trip_service_dict, trip_route_dict,
                 trips_df, calendar_df, calendar_dates_df) = reloaded

            records = collect_data(
                stops_df, stop_times_by_trip, route_stop_index,
                trip_suffix_index, trip_prefix_index, trip_service_dict, trip_route_dict,
                trips_df, calendar_df, calendar_dates_df, conn
            )

            elapsed = time.time() - t_start
            print(f"Ciclo completado en {elapsed:.1f}s | {records} registros guardados")

            next_time = datetime.now() + timedelta(seconds=COLLECTION_INTERVAL)
            print(f"Proxima recoleccion: {next_time.strftime('%H:%M:%S')} (en {COLLECTION_INTERVAL}s)")
            time.sleep(COLLECTION_INTERVAL)

        except KeyboardInterrupt:
            print("Interrumpido")
            break
        except Exception as e:
            print(f"Error inesperado: {e}")
            import traceback
            traceback.print_exc()
            print(f"Reintentando en {COLLECTION_INTERVAL}s...")
            time.sleep(COLLECTION_INTERVAL)

    if conn:
        conn.close()
        print("Conexion BD cerrada")


if __name__ == '__main__':
    main()
