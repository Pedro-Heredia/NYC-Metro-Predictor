#Recolector multi-lineas MTA 

import requests
import pandas as pd
from google.transit import gtfs_realtime_pb2
from datetime import datetime, timedelta
import psycopg2
import time
import pytz
from pathlib import Path



FEEDS = {
    '1-2-3-4-5-6-7': 'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs',
    'G': 'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-g',
    'L': 'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-l'
}

TARGET_ROUTES = {'1', '2', '3', '4', '5', '6', '7', 'G', 'L'}
TRAINS_PER_LINE = 4  # trenes por linea a mostrar en la terminal
COLLECTION_INTERVAL = 1500  # intervalo de minutos de cada recoleccion de datos
DEBUG_MODE = False  # false= guarda en base de datos
GTFS_DIR = Path(".")

DB_CONFIG = {
    "host": "localhost",
    "dbname": "delays2",
    "user": "postgres",
    "password": "tfg"
}


#FUNCIONES AUXILIARES

def get_ny_time():
    tz_ny = pytz.timezone("America/New_York")
    return datetime.now(tz_ny)

def load_gtfs_static():
    #Carga archivos GTFS incluyendo calendar_dates
    try:
        stops_df = pd.read_csv(GTFS_DIR / "stops.txt")
        stop_times_df = pd.read_csv(GTFS_DIR / "stop_times.txt")
        trips_df = pd.read_csv(GTFS_DIR / "trips.txt")
        calendar_df = pd.read_csv(GTFS_DIR / "calendar.txt")
        
        try:
            calendar_dates_df = pd.read_csv(GTFS_DIR / "calendar_dates.txt")
        except:
            calendar_dates_df = pd.DataFrame()
        
        
        return stops_df, stop_times_df, trips_df, calendar_df, calendar_dates_df
    except FileNotFoundError as e:
        print(f"Error cargando datos: {e}")
        return None, None, None, None, None

def time_to_minutes(time_str):
    #Convierte HH:MM:SS a minutos
    if pd.isna(time_str) or time_str == '':
        return None
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    return hours * 60 + minutes

def get_stop_name(stop_id, stops_df):
    #Obtiene nombre de parada (limpiando el ID base sin sufijos)
    # Limpiar el stop_id de sufijos (N, S, etc.)
    base_stop_id = stop_id.rstrip('NS')
    
    stop_row = stops_df[stops_df['stop_id'] == stop_id]
    if not stop_row.empty:
        return stop_row.iloc[0]['stop_name']
    
    # Si no encuentra, intentar con el ID base
    stop_row = stops_df[stops_df['stop_id'] == base_stop_id]
    if not stop_row.empty:
        return stop_row.iloc[0]['stop_name']
    
    return stop_id

def calculate_delay_minutes(scheduled_time_str, estimated_datetime):
    #Calcula el retraso en minutos
    if not scheduled_time_str:
        return None
    
    scheduled_minutes = time_to_minutes(scheduled_time_str)
    estimated_minutes = estimated_datetime.hour * 60 + estimated_datetime.minute
    
    if not scheduled_minutes:
        return None
    
    delay = estimated_minutes - scheduled_minutes
    
    # Manejo cruce de medianoche
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
    #Obtiene service_ids activos con calendar_dates
    today = datetime.now()
    weekday = today.weekday()
    today_str = today.strftime('%Y%m%d')
    
    weekday_map = {
        0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday',
        4: 'friday', 5: 'saturday', 6: 'sunday'
    }
    
    current_day = weekday_map[weekday]
    active_services = set()
    
    # Servicios regulares
    for _, service in calendar_df.iterrows():
        if service[current_day] == 1:
            active_services.add(service['service_id'])
    
    # Excepciones
    if not calendar_dates_df.empty:
        today_exceptions = calendar_dates_df[calendar_dates_df['date'] == int(today_str)]
        
        for _, exception in today_exceptions.iterrows():
            if exception['exception_type'] == 1:
                active_services.add(exception['service_id'])
            elif exception['exception_type'] == 2:
                active_services.discard(exception['service_id'])
    
    return list(active_services)

def find_scheduled_trip_debug(future_stops, route_id, direction_hint, trips_df, stop_times_df, active_services, train_id):
    # Encuentra el viaje programado 
    print(f"\n-Buscando scheduled trip para tren {train_id}")
    print(f"Route: {route_id}, Direction hint: {direction_hint}")
    print(f"Paradas a matchear: {len(future_stops)}")
    
    # 1. Filtrar trips activos de esta ruta
    route_trips = trips_df[
        (trips_df['route_id'] == route_id) & 
        (trips_df['service_id'].isin(active_services))
    ]
    
    if direction_hint:
        direction_id = 1 if 'S' in direction_hint else 0
        route_trips = route_trips[route_trips['direction_id'] == direction_id]
    
    print(f"Trips candidatos (con service_id): {len(route_trips)}")
    
    # 2. Si no hay trips con service_ids activos, buscar cualquier trip de la ruta
    if route_trips.empty:
        print(f"No trips con service_id activo, buscando cualquier trip de la ruta")
        route_trips = trips_df[trips_df['route_id'] == route_id]
        if direction_hint:
            direction_id = 1 if 'S' in direction_hint else 0
            route_trips = route_trips[route_trips['direction_id'] == direction_id]
        print(f"         Trips candidatos (sin filtro service_id): {len(route_trips)}")
    
    trip_ids = route_trips['trip_id'].tolist()
    
    if not trip_ids:
        print(f"No se encontraron trips candidatos")
        return None
    
    # 3. Buscar el mejor match
    best_trip_id = None
    best_score = float('inf')
    
    # Solo comparar las primeras 5 paradas
    stops_to_check = min(5, len(future_stops))
    print(f"Comparando con primeras {stops_to_check} paradas...")
    print(f"Paradas reales: {[s['stop_id'] for s in future_stops[:stops_to_check]]}")
    
    trips_checked = 0
    for trip_id in trip_ids:
        scheduled_stops = stop_times_df[stop_times_df['trip_id'] == trip_id]
        
        if scheduled_stops.empty:
            continue
        
        # Intentar validar horario con la primera parada que coincida, para intentar coger bien los trenes express
        first_matching_scheduled_time = None
        first_matching_real_time = None
        
        for stop_data in future_stops[:stops_to_check]:
            stop_id = stop_data['stop_id']
            
            # Buscar esta parada en scheduled
            scheduled_stop = scheduled_stops[scheduled_stops['stop_id'] == stop_id]
            
            if scheduled_stop.empty:
                base_stop_id = stop_id.rstrip('NS')
                scheduled_stop = scheduled_stops[scheduled_stops['stop_id'] == base_stop_id]
            
            if not scheduled_stop.empty:
                first_matching_scheduled_time = scheduled_stop.iloc[0]['arrival_time']
                first_matching_real_time = stop_data['arrival_time']
                break
        
        # Si HAY parada comun, validar que la hora sea razonable
        if first_matching_scheduled_time:
            scheduled_minutes = time_to_minutes(first_matching_scheduled_time)
            real_minutes = first_matching_real_time.hour * 60 + first_matching_real_time.minute
            
            # Ajustar por cruce de medianoche
            if scheduled_minutes and scheduled_minutes < 360 and real_minutes > 1200:
                scheduled_minutes += 1440
            
            if scheduled_minutes:
                time_diff = abs(real_minutes - scheduled_minutes)
                
                # Manejar cruce de medianoche
                time_diff = min(
                    time_diff,
                    abs((real_minutes + 1440) - scheduled_minutes),
                    abs(real_minutes - (scheduled_minutes + 1440))
                )
                
                # Si el trip scheduled esta a mas de 30 min del real, descartarlo
                if time_diff > 30:
                    continue
        
        # Si NO hay parada comun en las primeras 5, dejar que pase
        # (sera evaluado despues por el matching de paradas)
            
        trips_checked += 1
        
        total_diff = 0
        matched_stops = 0
        
        for stop_data in future_stops[:stops_to_check]:
            stop_id = stop_data['stop_id']
            real_time = stop_data['arrival_time']
            
            # Buscar coincidencia en scheduled_stops
            scheduled_stop = scheduled_stops[scheduled_stops['stop_id'] == stop_id]
            
            # Si no encuentra, intentar sin sufijo direccional (ej: '235N' -> '235')
            if scheduled_stop.empty:
                base_stop_id = stop_id.rstrip('NS')
                scheduled_stop = scheduled_stops[scheduled_stops['stop_id'] == base_stop_id]
            
            # Si la parada no esta en el horario, es un tren express para esa parada
            if scheduled_stop.empty:
                continue 
            
            # Obtener tiempos y normalizar a minutos
            scheduled_time_str = scheduled_stop.iloc[0]['arrival_time']
            scheduled_minutes = time_to_minutes(scheduled_time_str)
            real_minutes = real_time.hour * 60 + real_time.minute
            
            # Manejo de cambio de dia (medianoche)
            if scheduled_minutes and scheduled_minutes < 360 and real_minutes > 1200:
                scheduled_minutes += 1440
            
            if scheduled_minutes is not None:
                diff = abs(real_minutes - scheduled_minutes)
                actual_diff = min(
                    diff,
                    abs((real_minutes + 1440) - scheduled_minutes),
                    abs(real_minutes - (scheduled_minutes + 1440))
                )
                
                total_diff += actual_diff
                matched_stops += 1
        
        # 4. Evaluacion del Score
        if matched_stops > 0:
            avg_diff = total_diff / matched_stops
            # Penalizacion por paradas faltantes (si el trip real tiene paradas que el schedule no)
            coverage_penalty = (stops_to_check - matched_stops) * 30
            final_score = avg_diff + coverage_penalty
            
            if final_score < best_score:
                best_score = final_score
                best_trip_id = trip_id
                
                # Si el match es muy preciso (menos de 5 min de error acumulado), terminar busqueda
                if final_score < 5:
                    print(f"Match perfecto encontrado: {trip_id}")
                    break
    
    print(f"Trips evaluados: {trips_checked}")
    if best_trip_id:
        print(f"Mejor match: {best_trip_id} (score: {best_score:.1f})")
    else:
        print(f"No se encontro match adecuado")
    
    return best_trip_id

def get_scheduled_times(trip_id, stop_ids, stop_times_df):
    #Obtiene horarios programados (manejando stop_ids con/sin sufijos)
    scheduled_times = {}
    
    if not trip_id:
        return scheduled_times
    
    trip_stops = stop_times_df[stop_times_df['trip_id'] == trip_id]
    
    for stop_id in stop_ids:
        # Intentar match exacto
        stop_time = trip_stops[trip_stops['stop_id'] == stop_id]
        
        # Si no encuentra, intentar sin sufijo direccional
        if stop_time.empty:
            base_stop_id = stop_id.rstrip('NS')
            stop_time = trip_stops[trip_stops['stop_id'] == base_stop_id]
        
        if not stop_time.empty:
            scheduled_times[stop_id] = stop_time.iloc[0]['arrival_time']
    
    return scheduled_times

def calculate_cumulative_delay(future_stops, scheduled_times):
    #Calcula retraso acumulado
    cumulative_delays = {}
    
    for i, stop_data in enumerate(future_stops):
        stop_id = stop_data['stop_id']
        arrival_time = stop_data['arrival_time']
        scheduled_time_str = scheduled_times.get(stop_id, None)
        
        if scheduled_time_str:
            delay_minutes = calculate_delay_minutes(scheduled_time_str, arrival_time)
            if delay_minutes is not None:
                if i == 0:
                    cumulative_delays[stop_id] = delay_minutes
                else:
                    prev_cumulative = list(cumulative_delays.values())[-1] if cumulative_delays else 0
                    cumulative_delays[stop_id] = max(prev_cumulative, delay_minutes) if prev_cumulative is not None else delay_minutes
            else:
                cumulative_delays[stop_id] = None
        else:
            cumulative_delays[stop_id] = None
    
    return cumulative_delays

def normalize_time_format(time_str):
    # 24+ horas a formato normal
    if not time_str or time_str == "N/A":
        return time_str
    
    parts = time_str.split(':')
    if len(parts) >= 2:
        hours = int(parts[0])
        if hours >= 24:
            hours = hours - 24
        return f"{hours:02d}:{parts[1]}:{parts[2] if len(parts) > 2 else '00'}"
    
    return time_str

def is_peak_hour(ny_time):
    #Determina si es hora pico
    hour = ny_time.hour
    minute = ny_time.minute
    current_minutes = hour * 60 + minute
    
    peak_ranges = [
        (7 * 60, 9 * 60),
        (13 * 60, 15 * 60),
        (17 * 60, 18 * 60),
    ]
    
    return any(start <= current_minutes <= end for start, end in peak_ranges)


#Funcion principal - trenes por lineas

def collect_and_display_debug(stops_df, stop_times_df, trips_df, calendar_df, calendar_dates_df, conn=None):
    #Muestra TRAINS_PER_LINE trenes de cada loinea
    
    current_time = datetime.now()
    ny_time = get_ny_time()
    active_services = get_current_service_ids(calendar_df, calendar_dates_df)
    
    print(f"\n")
    print(f"Hora NYC: {ny_time.strftime('%H:%M:%S')}")
    print(f"Servicios activos: {active_services}")
    
    
    trains_by_route = {}  # {route_id: [train1, train2, train3]}
    all_data = []
    
    # Procesar feeds
    for feed_name, feed_url in FEEDS.items():
        print(f"\tConsultando feed: {feed_name}")
        
        try:
            response = requests.get(feed_url, timeout=30)
            
            if response.status_code != 200:
                print(f"Error {response.status_code}")
                continue
            
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(response.content)
            
            print(f"{len(feed.entity)} entidades recibidas")
            
            # Procesar entidades
            for entity in feed.entity:
                if not entity.HasField('trip_update'):
                    continue
                
                trip = entity.trip_update
                trip_id = trip.trip.trip_id
                route_id = trip.trip.route_id
                
                # Filtrar lineas objetivo
                if route_id not in TARGET_ROUTES:
                    continue
                
                # Inicializar lista para esta linea
                if route_id not in trains_by_route:
                    trains_by_route[route_id] = []
                
                # Ya tenemos suficientes trenes de esta linea?
                if len(trains_by_route[route_id]) >= TRAINS_PER_LINE:
                    continue
                
                # Determinar direccin
                direction = "DESCONOCIDA"
                direction_hint = None
                if '..' in trip_id:
                    direction_hint = trip_id.split('..')[1][0] if len(trip_id.split('..')[1]) > 0 else None
                    if direction_hint == 'S':
                        direction = "SOUTH"
                    elif direction_hint == 'N':
                        direction = "NORTH"
                
                # Recopilar paradas futuras
                future_stops = []
                
                if trip.stop_time_update:
                    for stop_update in trip.stop_time_update:
                        if stop_update.HasField('arrival'):
                            arrival_time = datetime.fromtimestamp(stop_update.arrival.time)
                            
                            if arrival_time > (current_time - timedelta(minutes=1)):
                                minutes_until = (arrival_time - current_time).total_seconds() / 60
                                future_stops.append({
                                    'stop_id': stop_update.stop_id,
                                    'arrival_time': arrival_time,
                                    'minutes_until': minutes_until
                                })
                
                if not future_stops:
                    continue
                
                future_stops.sort(key=lambda x: x['minutes_until'])
                
                #Buscar scheduled_trip para este tren especifico
                best_trip = find_scheduled_trip_debug(
                    future_stops, route_id, direction_hint,
                    trips_df, stop_times_df, active_services, trip_id
                )
                
                # FILTRO 1: Si no se encuentra scheduled trip, saltar este tren
                if not best_trip:
                    print(f"TREN DESCARTADO: No se encontro su scheduled trip")
                    continue
                
                # Obtener horarios programados
                stop_ids = [stop['stop_id'] for stop in future_stops]
                scheduled_times = get_scheduled_times(best_trip, stop_ids, stop_times_df)
                
                # FILTRO 2: Al menos 60% de paradas deben tener scheduled_time
                valid_scheduled = sum(1 for v in scheduled_times.values() if v is not None)
                coverage_ratio = valid_scheduled / len(future_stops) if future_stops else 0
                
                if coverage_ratio < 0.6:
                    print(f"TREN DESCARTADO: Solo {valid_scheduled}/{len(future_stops)} paradas con horario ({coverage_ratio*100:.0f}%)")
                    continue
                
                # Calcular retrasos acumulados
                cumulative_delays = calculate_cumulative_delay(future_stops, scheduled_times)
                
                # FILTRO 3: Las primeras 5 paradas deben tener scheduled_time y delay razonable
                train_has_invalid_delays = False
                valid_delays_in_first_stops = 0
                
                for i, stop_data in enumerate(future_stops[:5]):
                    stop_id = stop_data['stop_id']
                    arrival_time = stop_data['arrival_time']
                    scheduled_time_str = scheduled_times.get(stop_id, None)
                    
                    if not scheduled_time_str:
                        # Si alguna de las primeras 5 no tiene scheduled_time, se descarta
                        print(f"TREN DESCARTADO: Parada {i+1} ({get_stop_name(stop_id, stops_df)}) sin scheduled_time")
                        train_has_invalid_delays = True
                        break
                    
                    delay = calculate_delay_minutes(scheduled_time_str, arrival_time)
                    
                    if delay is None:
                        print(f"TREN DESCARTADO: No se pudo calcular delay en parada {i+1}") #para siguiente parada
                        train_has_invalid_delays = True
                        break
                    
                    
                    
                    valid_delays_in_first_stops += 1
                
                if train_has_invalid_delays or valid_delays_in_first_stops < 5:
                    continue
                
                # Guardar info de es tren
                trains_by_route[route_id].append({
                    'trip_id': trip_id,
                    'direction': direction,
                    'direction_hint': direction_hint,
                    'future_stops': future_stops,
                    'best_trip': best_trip,
                    'scheduled_times': scheduled_times,
                    'cumulative_delays': cumulative_delays
                })
                
                # para uardar datos en la BD
                dow = current_time.weekday()
                hour = ny_time.hour
                is_peak = is_peak_hour(ny_time)
                
                for stop_data in future_stops:
                    stop_id = stop_data['stop_id']
                    arrival_time = stop_data['arrival_time']
                    
                    stop_name = get_stop_name(stop_id, stops_df)
                    scheduled_time_str = scheduled_times.get(stop_id, None)
                    
                    delay_minutes_raw = None
                    if scheduled_time_str:
                        delay_minutes_raw = calculate_delay_minutes(scheduled_time_str, arrival_time)
                    
                    cumulative_delay = cumulative_delays.get(stop_id, None)
                    
                    row_data = {
                        'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        'train_id': trip_id,
                        'route_id': route_id,
                        'direction': direction,
                        'stop_id': stop_id,
                        'stop_name': stop_name,
                        'scheduled_time': scheduled_time_str,
                        'estimated_time': arrival_time.strftime("%H:%M:%S"),
                        'delay_minutes': round(delay_minutes_raw, 1) if delay_minutes_raw is not None else None,
                        'dow': dow,
                        'hour': hour,
                        'is_peak_hour': is_peak,
                        'cumulative_delay': round(cumulative_delay, 1) if cumulative_delay is not None else None
                    }
                    
                    all_data.append(row_data)
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Mostrar resultados
    print(f"\n")
    print(f"Resultados: {TRAINS_PER_LINE} Tenes por linea")
    print(f"\n")
    print(f"Lineas totales: {len(trains_by_route)}")
    
    dow = current_time.weekday()
    hour = ny_time.hour
    is_peak = is_peak_hour(ny_time)
    
    for route_id in sorted(trains_by_route.keys()):
        trains = trains_by_route[route_id]
        
        
        print(f"\n\tLinea {route_id} - {len(trains)} trenes")
        
        
        for train_idx, train_info in enumerate(trains, 1):
            trip_id = train_info['trip_id']
            direction = train_info['direction']
            future_stops = train_info['future_stops']
            best_trip = train_info['best_trip']
            scheduled_times = train_info['scheduled_times']
            cumulative_delays = train_info['cumulative_delays']
            
           
            print(f"\nTREN #{train_idx} - {trip_id}")
            
            print(f"Direccion: {direction}")
            
            if best_trip:
                print(f"Scheduled trip:  {best_trip}")
            else:
                print(f"Scheduled trip:  NO ENCONTRADO")
            
            print(f"Scheduled times encontrados: {len([v for v in scheduled_times.values() if v])} de {len(future_stops[:10])}")
            
            series_id = f"{route_id}_{direction}"
            
            print(f"Series ID: {series_id} | DOW: {dow} | Hora: {hour} | Peak: {'Si' if is_peak else 'No'}")
            print(f"{'#':<3} {'Parada':<30} {'Est.':<8} {'Prog.':<8} {'Retraso':<8} {'Acum.'}")
            
            
            for i, stop_data in enumerate(future_stops[:10], 1):
                stop_id = stop_data['stop_id']
                arrival_time = stop_data['arrival_time']
                
                stop_name = get_stop_name(stop_id, stops_df)
                scheduled_time_str = scheduled_times.get(stop_id, None)
                
                delay_str = "N/A"
                if scheduled_time_str:
                    delay_minutes_raw = calculate_delay_minutes(scheduled_time_str, arrival_time)
                    if delay_minutes_raw is not None:
                        delay_seconds = delay_minutes_raw * 60
                        
                        if abs(delay_seconds) < 60:
                            delay_str = "PUNTUAL"
                        elif delay_seconds >= 60:
                            delay_min = delay_seconds // 60
                            delay_str = f"+{delay_min:.0f}m"
                        else:
                            delay_min = abs(delay_seconds) // 60
                            delay_str = f"-{delay_min:.0f}m"
                
                estimated_str = arrival_time.strftime("%H:%M")
                scheduled_str = normalize_time_format(scheduled_time_str)[:5] if scheduled_time_str else "N/A"
                
                cumulative_delay = cumulative_delays.get(stop_id, None)
                cumulative_str = f"{cumulative_delay:+.0f}m" if cumulative_delay is not None else "N/A"
                
                print(f"{i:<3} {stop_name[:29]:<30} {estimated_str:<8} {scheduled_str:<8} {delay_str:<8} {cumulative_str}")
    
    
    print(f"Total registros generados: {len(all_data)}")
    
    
    # Guardar en BD si no esta en debug
    if not DEBUG_MODE and conn is not None:
        try:
            cur = conn.cursor()
            
            for row in all_data:
                cur.execute("""
                    INSERT INTO delays (
                        timestamp, train_id, route_id, direction,
                        stop_id, stop_name, scheduled_time, estimated_time,
                        delay_minutes, dow, hour, is_peak_hour, cumulative_delay
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    row['timestamp'], row['train_id'], row['route_id'], row['direction'],
                    row['stop_id'], row['stop_name'], row['scheduled_time'], row['estimated_time'],
                    row['delay_minutes'], row['dow'], row['hour'], row['is_peak_hour'],
                    row['cumulative_delay']
                ))
            
            conn.commit()
            cur.close()
            print(f"Datos guardados en BD")
            
        except Exception as e:
            print(f"Error guardando en BD: {e}")
            conn.rollback()
    
    return len(all_data)

#Funcion principal (main)


def main():
    print("\n")
    print(f"Mostrando {TRAINS_PER_LINE} trenes por linea")
    print(f"Intervalo de recoleccion: {COLLECTION_INTERVAL // 60} minutos")
    
    
    # Cargar GTFS
    stops_df, stop_times_df, trips_df, calendar_df, calendar_dates_df = load_gtfs_static()
    
    if stops_df is None:
        print("\nNo se pudieron cargar archivos GTFS")
        return
    
    # Conectar a BD
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print(f"Conectado a Postgre: {DB_CONFIG['dbname']}")
    except Exception as e:
        print(f"Error conectando a BD: {e}")
        return
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            print(f"\n")
            print(f"Iteracion #{iteration}")
            
            
            # Recolectar datos
            records = collect_and_display_debug(
                stops_df, stop_times_df, trips_df, calendar_df, calendar_dates_df, conn
            )
            
            print(f"\nRecoleccion completada: {records} registros guardados")
            
            # Siguiente recoleccion
            next_collection = datetime.now() + timedelta(seconds=COLLECTION_INTERVAL)
            print(f"\nProxima recoleccion: {next_collection.strftime('%H:%M:%S')}")
            print(f"Esperando {COLLECTION_INTERVAL // 60} minutos...")
            
            
            time.sleep(COLLECTION_INTERVAL)
            
        except KeyboardInterrupt:
            print("\nRecoleccion interrumpida")
            break
        except Exception as e:
            print(f"\nError inesperado: {e}")
            import traceback
            traceback.print_exc()
            print(f"Reintentando en {COLLECTION_INTERVAL // 60} minutos...")
            time.sleep(COLLECTION_INTERVAL)
    
    # Cerrar conexion
    if conn:
        conn.close()
        print("\nConexion a BD cerrada")
    
    

if __name__ == '__main__':
    main()