
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from service_types import should_train_stop_here, STOP_SERVICE_TYPES, FULL_TIME_ONLY_ROUTES, get_stop_service_type
import math
warnings.filterwarnings('ignore')

#https://docs.python.org/3/library/pickle.html
#https://networkx.org/documentation/stable/tutorial.html    dijkstra
#https://xgboost.readthedocs.io/en/stable/parameter.html

GTFS_DIR = Path(".")
OVERHEAD_ESTACION = 2.0
TIEMPO_TRANSBORDO = 3.0
LINEAS_VALIDAS = {'1', '2', '3', '4', '5', '6', '7', 'A', 'C', 'E', 'B', 'D', 'F', 'M', 'J', 'Z', 'N', 'Q', 'R', 'W', 'G', 'L'}


try:
    with open('modelo_xgboost_final.pkl', 'rb') as f:
        modelo_data = pickle.load(f)
    
    model = modelo_data['model']
    scaler = modelo_data.get('scaler', None)
    encoders = modelo_data['encoders']
    features = modelo_data['features']
    mae_test = modelo_data['mae_test']
    
    print(f"MAE del modelo: {mae_test:.3f} min")
    
except FileNotFoundError:
    print("ERROR: No se encontro modelo_xgboost_final.pkl")
    exit(1)

#para cargar dataset procesado


try:
    viajes_df = pd.read_pickle('dataset_viajes_raw.pkl')
    print(f"Dataset cargado: {len(viajes_df):,} registros")
except FileNotFoundError:
    print("ERROR: No se encontro dataset_viajes_raw.pkl")
    exit(1)

#cargar GTFS solo horarios



DIAS = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']

try:
    stop_times_df = pd.read_csv(GTFS_DIR / "stop_times.txt")
    trips_df = pd.read_csv(GTFS_DIR / "trips.txt")
    stops_df = pd.read_csv(GTFS_DIR / "stops.txt")  
    print("GTFS cargado correctamente")
except Exception as e:
    print(f"Error cargando archivos GTFS: {e}")
    exit(1)

def parsear_fecha(fecha_str):
    try:
        fecha = datetime.strptime(fecha_str, "%d/%m/%Y")
        dow   = fecha.weekday()  # 0=lunes, 6=domingo, calculado automaticamente
        return fecha, dow
    except ValueError:
        return None, None

def limpiar_stop_id(stop_id):
    return stop_id.rstrip('NS')

def obtener_hora_actual():
    ahora = datetime.now()
    return f"{ahora.hour:02d}:{ahora.minute:02d}", ahora.weekday()

def time_str_to_minutes(time_str):
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])


#clasificacion trips segun su tipo de de servicio

# trip_ids_nocturnos: trips que incluyen paradas night_service -> solo validos de 22h a 6h
# trip_ids_rush: trips que incluyen paradas rush_hour_only --> solo en pico laboral
# trip_ids_part_time: trips que incluyen paradas part_time -> validos de 6h a 23h
# el resto son full_time, siempre paran en las paradas



trip_ids_nocturnos  = set()
trip_ids_rush       = set()
trip_ids_part_time  = set()

# lookup rapido: base_stop_id -> {route_id -> service_type}
stop_service_lookup = {}

for (route_id, stop_id), stype in STOP_SERVICE_TYPES.items(): # pasar de lista  a dic par poder coger el tipo de servicio  por [parada][linea]
    base = stop_id.rstrip('NS') 
    if base not in stop_service_lookup:
        stop_service_lookup[base] = {}
    stop_service_lookup[base][route_id] = stype 

for route_id in LINEAS_VALIDAS: 
    if route_id in FULL_TIME_ONLY_ROUTES:  #ignorar las lineas normales
        continue
   
    trips_de_ruta = trips_df[trips_df['route_id'] == route_id]['trip_id'].unique() #obtenemos todos los IDS unicos del viaje para la linea actual
    if len(trips_de_ruta) == 0:
        continue

    st_ruta = stop_times_df[stop_times_df['trip_id'].isin(trips_de_ruta)].copy() # filtrar horarios solo para los viajes de esta ruta
    st_ruta['stop_base'] = st_ruta['stop_id'].apply(limpiar_stop_id)

    #analizar cada viaje para ver en que paradas se pararia
    for trip_id, group in st_ruta.groupby('trip_id'):
        for base_stop in group['stop_base'].unique():
            #verificar si xada parada actual tiene una restriccion
            if base_stop in stop_service_lookup and route_id in stop_service_lookup[base_stop]:
                stype = stop_service_lookup[base_stop][route_id]
                
                if stype == 'night_service':
                    trip_ids_nocturnos.add(trip_id) 
                    break #prioridad, si tiene una parada nocturna, el viaje es nocturno
                
                elif stype == 'rush_hour_only':
                    trip_ids_rush.add(trip_id) 
                    break
                
                elif stype == 'part_time':
                    trip_ids_part_time.add(trip_id) 
                    # sin break ya que un viaje part-time puede ser tambn nocturno en otra parada


print(f"Numeros de viajes; nocturnos: {len(trip_ids_nocturnos)} | rush: {len(trip_ids_rush)} | part-time: {len(trip_ids_part_time)}")


#filtro de seguridad para que los trenes cogidos sean validos
def trips_validos_para_hora(trips_candidatos, hour, dow):
    es_nocturno  = (hour < 6 or hour >= 22)
    es_laborable = (dow < 5)
    es_rush      = es_laborable and ((6.5 <= hour < 9.5) or (15.5 <= hour < 20.0))
    es_part_time = (6 <= hour < 23)  # part_time: disponible de 6h a 23h
   #part-time: el tren parara en esa estacion siempre que no sea de noche
    validos = []
    for t in trips_candidatos:
        if t in trip_ids_nocturnos:
            if es_nocturno:
                validos.append(t)
        elif t in trip_ids_rush:
            if es_rush:
                validos.append(t)
        elif t in trip_ids_part_time:
            if es_part_time:
                validos.append(t)
        else:
            validos.append(t)  # full_time: siempre
    return validos


# construccion de diferentes grafos para usar en funcion de la hora del dia que sea
# G_METRO_DIA   -> servicio normal (6h-22h laborable)
# G_METRO_NOCHE -> servicio nocturno (22h-6h): L2 para en 86 St, 79 St, etc.
# G_METRO_RUSH  -> horas pico laborables: L5 llega al norte del Bronx, etc.

G_METRO_DIA   = nx.DiGraph()
G_METRO_NOCHE = nx.DiGraph()
G_METRO_RUSH  = nx.DiGraph()


def construir_un_grafo(grafo, trips_excluidos, trips_requeridos=None):
    grafo.clear()
    rutas_dir = trips_df[['route_id', 'direction_id']].drop_duplicates()
    count_viajes = 0

    for _, row in rutas_dir.iterrows():
        r_id = str(row['route_id'])
        if r_id not in LINEAS_VALIDAS: # Solo procesar las lineas que nos interesan (1-7, G, L)
            continue

        d_id = row['direction_id'] 
        todos = trips_df[       # Buscar todos los viajes que hacen esta ruta en este sentido
            (trips_df['route_id'] == r_id) & (trips_df['direction_id'] == d_id)
        ]['trip_id'].tolist()
        if not todos:
            continue

        # Aplicar filtros
        candidatos = [t for t in todos if t not in trips_excluidos] #Quitar viajes que no tocan (ej no usar nocturnos en el mapa de dia)
        if trips_requeridos is not None:
            #Si es mapa rush, priorizar trenes de hora punta pero si no hay, usamos los full-time
            rush_cands = [t for t in candidatos if t in trips_requeridos]
            candidatos = rush_cands if rush_cands else candidatos

        if not candidatos:
            candidatos = todos #si los filtros vacian la lista, usar todos

        # Elegir el trip representativo (el mas frecuente en numero de paradas)
        trips_muestra = pd.Series(candidatos).head(200) 
        stop_counts = stop_times_df[
            stop_times_df['trip_id'].isin(trips_muestra)
        ].groupby('trip_id').size() # lo normal es que en la linea X haya Y numero de paradas, para ignorar trenes raros (averias y tal)

        if stop_counts.empty:
            continue

        mode_length = stop_counts.mode()[0] #para saber cuantas paradas tiene el trayecto "comun"
        best_trip = stop_counts[stop_counts == mode_length].index[0] #elegimos el primer viaje que coincida con la longitud 

        # DEBUG para lo de la parte de la linea de rush hour que va por otro lado (SOLUCIONADO, ERA QUE HABIA PARADAS "DUPLICADAS" POR MISMA AVENIDA EN EL ARCHIVO DE TIPOS DE PARADAS)
        if r_id == '5':
            #print(f"DEBUG  Trip representativo linea 5: {best_trip}, paradas: {mode_length}")
            stops_seq_debug = stop_times_df[
                stop_times_df['trip_id'] == best_trip
            ].sort_values('stop_sequence')
            #print(stops_seq_debug['stop_id'].tolist())


        #sacar la secuencia de paradas del viaje "comun"
        stops_seq = stop_times_df[stop_times_df['trip_id'] == best_trip].sort_values('stop_sequence')
        lista_paradas = stops_seq['stop_id'].apply(limpiar_stop_id).tolist()

        for i in range(len(lista_paradas) - 1): # Creamos las aristas del viaje
            orig = (lista_paradas[i],   r_id)
            dest = (lista_paradas[i+1], r_id)
            grafo.add_edge(orig, dest, weight=1.5, type='viaje') 
            grafo.add_edge(dest, orig, weight=1.5, type='viaje') #permitir ambos sentidos
            count_viajes += 2 #uno ida otro vuelta

    # logica para los transbordos
    def haversine(lon1, lat1, lon2, lat2): #para solucionar el  problema de que en una misma calle hay 24 paradas con el mismo nombre (cambio de linea 2 a 6 en la 96 St)
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return 2 * math.asin(math.sqrt(a)) * 6371

    mapeo_nombres = {} 
    for n in grafo.nodes():
        stop_id, _ = n
        match = stops_df[stops_df['stop_id'] == stop_id]
        
        if not match.empty:
            nombre = match['stop_name'].iloc[0]
            lat = match['stop_lat'].iloc[0]
            lon = match['stop_lon'].iloc[0]
        else:
            nombre = stop_id
            lat, lon = 0, 0
            
        if nombre not in mapeo_nombres:
            mapeo_nombres[nombre] = []
        mapeo_nombres[nombre].append({'nodo': n, 'lat': lat, 'lon': lon}) #creamos un dic con los nombres de la parada y sus coord

    count_transbordos = 0
    for nombre, datos_nodos in mapeo_nombres.items():
        if len(datos_nodos) > 1:
            for d1 in datos_nodos:
                for d2 in datos_nodos:
                    if d1['nodo'] != d2['nodo']:
                        # verificamos la distancia fisica
                        if d1['lat'] != 0 and d2['lat'] != 0:
                            dist_km = haversine(d1['lon'], d1['lat'], d2['lon'], d2['lat'])
                            # Si estan a mas de 400 metros (0.4 km), son estaciones distintas en diferentes locations
                            if dist_km > 0.4:
                                continue
                                
                        grafo.add_edge(d1['nodo'], d2['nodo'], weight=100.0, type='transbordo')
                        count_transbordos += 1

    return count_viajes, count_transbordos



def construir_grafos(): #para solucionar el problema de no dar los viajes correctos segun la hora del dia que es
    cv, ct = construir_un_grafo(
        G_METRO_DIA, #part-time: tren 0parara en esta estacion siempre que no sea de noche ?????
        trips_excluidos=trip_ids_nocturnos | trip_ids_rush
    )
    print(f"dia-> {len(G_METRO_DIA.nodes())} nodos, {cv} tramos, {ct} transbordos")

    cv, ct = construir_un_grafo(
        G_METRO_NOCHE, #trip representativo a la noche, incluye todas las demas paradas
        trips_excluidos=trip_ids_rush,
        trips_requeridos=trip_ids_nocturnos
    )
    print(f"noche-> {len(G_METRO_NOCHE.nodes())} nodos, {cv} tramos, {ct} transbordos")

    cv, ct = construir_un_grafo(
        G_METRO_RUSH,
        trips_excluidos=trip_ids_nocturnos,
        trips_requeridos=trip_ids_rush
    )
    print(f"rush-> {len(G_METRO_RUSH.nodes())} nodos, {cv} tramos, {ct} transbordos")


construir_grafos()


def seleccionar_grafo(hour, dow):
    es_nocturno  = (hour < 6 or hour >= 22)
    es_laborable = (dow < 5)
    es_rush      = es_laborable and ((6.5 <= hour < 9.5) or (15.5 <= hour < 20.0))

    if es_nocturno:
        return G_METRO_NOCHE
    elif es_rush:
        return G_METRO_RUSH
    else:
        return G_METRO_DIA



def preparar_features_segmento(origen_id, destino_id, route_id, tiempo_total_min, dow,
                                travel_time, segment_num, total_segments,
                                delay_actual=None, cumulative_delay=None, fecha=None, direction_north=0):
    hour = int((tiempo_total_min // 60) % 24)
    minute_of_hour = int(tiempo_total_min % 60)
    minute_of_day = int(tiempo_total_min % 1440)

    fecha_ref = fecha if fecha is not None else datetime.now()
    month = fecha_ref.month
    is_weekend = 1 if dow >= 5 else 0

    orig_stype = get_stop_service_type(route_id, origen_id)
    dest_stype = get_stop_service_type(route_id, destino_id)

    viaje_data = {
        'origin_stop_id': origen_id,
        'destination_stop_id': destino_id,
        'route_id': route_id,
        'hour': hour,
        'day_of_week': dow,
        'travel_time_minutes': travel_time,
        'segment_number': segment_num,
        'total_segments': total_segments,
        'minute_of_hour': minute_of_hour,
        'minute_of_day': minute_of_day,
        'origin_is_part_time': 1 if orig_stype == 'part_time' else 0,
        'dest_is_part_time': 1 if dest_stype == 'part_time' else 0,
        'origin_is_rush_hour': 1 if orig_stype == 'rush_hour_only' else 0,
        'dest_is_rush_hour': 1 if dest_stype == 'rush_hour_only' else 0,
        'delay_at_origin': delay_actual if delay_actual is not None else np.nan,
        'cumulative_delay_origin': cumulative_delay if cumulative_delay is not None else np.nan,
        'month': month,
        'is_weekend': is_weekend,
        'direction_north': direction_north
    }

    df = pd.DataFrame([viaje_data])

    df['position_ratio'] = df['segment_number'] / df['total_segments']
    df['is_early_segment'] = (df['position_ratio'] < 0.33).astype(int)
    df['is_late_segment'] = (df['position_ratio'] > 0.67).astype(int)

    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)

    df['is_line_3'] = (df['route_id'] == '3').astype(int)
    df['is_line_5'] = (df['route_id'] == '5').astype(int)
    df['is_express'] = df['route_id'].isin(['3', '5', '7']).astype(int)

    df['hour_x_position'] = df['hour'] * df['position_ratio']
    df['rush_x_segment'] = (df['is_morning_rush'] + df['is_evening_rush']) * df['segment_number']

    for col in ['origin_stop_id', 'destination_stop_id', 'route_id']:
        if col in encoders:
            try:
                df[f'{col}_encoded'] = encoders[col].transform(df[col])
            except:
                df[f'{col}_encoded'] = 0

    X = df[features].values
    return scaler.transform(X) if scaler is not None else X



def predecir_ruta_completa(ruta_stops, hora_salida_min, dow, grafo=None, delay_actual=None):
    if grafo is None:
        grafo = G_METRO_DIA

    hora_actual = hora_salida_min
    delay_total = 0
    prog_total = 0

    r_id_tramo = "1"
    try:
        r_id_tramo = grafo.get_edge_data(ruta_stops[0], ruta_stops[1])['linea']
    except:
        pass

    # Modo tiempo real: delay_actual conocido, se propaga prediccion a prediccion.
    # Modo pre-viaje: delay_actual=None -> NaN en todos los segmentos.
    delay_en_origen = delay_actual
    cumul_delay = delay_actual  # None si pre-viaje
    fecha_actual = datetime.now()

    for i in range(len(ruta_stops) - 1):
        origen = ruta_stops[i]
        destino = ruta_stops[i+1]

        t_base = 2.0
        try:
            t_base = grafo.get_edge_data(origen, destino)['weight']
        except:
            pass

        X = preparar_features_segmento(
            origen, destino, r_id_tramo, hora_actual, dow,
            t_base, i+1, len(ruta_stops) - 1,
            delay_actual=delay_en_origen,
            cumulative_delay=cumul_delay,
            fecha=fecha_actual
        )
        delay = model.predict(X)[0]

        real = max(0.5, t_base + delay)
        delay_real = real - t_base

        hora_actual += real
        delay_total += delay_real
        prog_total += t_base

        # En modo tiempo real propagamos el delay predicho al siguiente segmento.
        if delay_actual is not None:
            delay_en_origen = delay
            cumul_delay = (cumul_delay or 0) + delay_real

    return {
        'hora_llegada_min': hora_actual,
        'delay_total': delay_total,
        'tiempo_prog': prog_total
    }, None

#tiempo programado histrico, XGBoost para predecir el delay, y avanza el reloj



def buscar_trenes_en_gtfs_rapido(route_id, stop_id_origen, hora_deseada_min, cantidad=10, hour=None, dow=None):
    #para tener los n trips mas cercanos a la hora deseada

    base_origen = limpiar_stop_id(stop_id_origen)
    trips_linea = trips_df[trips_df['route_id'] == route_id]['trip_id'].unique()  # todos los viajes de esa linea

    # Filtrar por tipo de servicio si se conoce la hora
    if hour is not None and dow is not None:
        trips_linea = trips_validos_para_hora(trips_linea, hour, dow) # (no usar nocturnos de dia ni rush en finde)

    if len(trips_linea) == 0:
        return [] 

    stop_times_filtrados = stop_times_df[
        (stop_times_df['stop_id'].str.replace('[NS]', '', regex=True) == base_origen) & # paradas que coincidan con el origen
        (stop_times_df['trip_id'].isin(trips_linea)) # solo de los trips validos para esa hora
    ].copy()

    if stop_times_filtrados.empty:
        return []

    stop_times_filtrados['hora_min'] = stop_times_filtrados['departure_time'].apply(time_str_to_minutes)
    stop_times_filtrados['diff'] = (stop_times_filtrados['hora_min'] - hora_deseada_min).abs()
    closest = stop_times_filtrados.nsmallest(cantidad, 'diff') # quedarse con los N mas cercanos a la hora deseada, 18:58 es mejor que 19:03

    trenes = []
    for _, row in closest.iterrows():
        trenes.append({
            'trip_id': row['trip_id'],
            'hora_salida_min': int(row['hora_min']),
            'hora_salida_prog': row['departure_time']
        })
    return trenes


def calcular_opciones_dijkstra(origenes, destinos, hora_salida_prog, dow, es_llegada=False, hora_objetivo_real=None, linea_origen=None, linea_destino=None, delay_actual=None):


    h_parts = hora_salida_prog.split(':')
    hora_target_min = int(h_parts[0]) * 60 + int(h_parts[1]) # hora de busqueda en minutos desde medianoche
    hora_para_comparar = hora_objetivo_real if (es_llegada and hora_objetivo_real) else hora_target_min  # en modo llegada, comparar contra el objetivo real (empezara a busca antes de la hora puesta)

    # Seleccionar el grafo correcto segun hora y dia
    hora_ref = hora_para_comparar if es_llegada else hora_target_min
    hour_ref = int(hora_ref // 60) % 24
    grafo = seleccionar_grafo(hour_ref, dow)

    ids_origen = [o[0] for o in origenes] #stopsids de la lista de tuplas (id,nombre)
    ids_destino = [d[0] for d in destinos] # destinos

    nodos_origen  = [n for n in grafo.nodes() if n[0] in ids_origen  and (linea_origen  is None or n[1] == linea_origen)] #buscamos los stops id en el grafo 
    #y linea origen para lo de que fulton st de la linea 2 se confunde con el de la linea g
    nodos_destino = [n for n in grafo.nodes() if n[0] in ids_destino]

    if not nodos_origen or not nodos_destino:
        #print(f"DEBUG IDs origen: {ids_origen} | encontrados: {nodos_origen}")
        #print(f"DEBUG IDs destino: {ids_destino} | encontrados: {nodos_destino}")
        return []

    # 1 DIJKSTRA
    mejor_camino = None
    min_weight = float('inf')
    
    for start in nodos_origen:
        for end in nodos_destino:
            try:
                w = nx.shortest_path_length(grafo, start, end, weight='weight')
                if w < min_weight:
                    min_weight = w
                    mejor_camino = nx.shortest_path(grafo, start, end, weight='weight')
            except:
                continue

    if not mejor_camino:
        return []

    # DEBUG 
    #print(f"\nDEBUG DIJKSTRA Camino elegido (peso total: {min_weight:.1f}):")
    '''for i in range(len(mejor_camino) - 1):
        n1, n2 = mejor_camino[i], mejor_camino[i + 1]
        ed = grafo.get_edge_data(n1, n2)

        nm1 = stops_df[stops_df['stop_id'] == n1[0]]['stop_name'].iloc[0] if not stops_df[stops_df['stop_id'] == n1[0]].empty else n1[0] #para coger el nombre de la parada (nodo1)

        nm2 = stops_df[stops_df['stop_id'] == n2[0]]['stop_name'].iloc[0] if not stops_df[stops_df['stop_id'] == n2[0]].empty else n2[0]

        if ed['type'] == 'transbordo':
            print(f"  {nm1} (L{n1[1]}) -> {nm2} (L{n2[1]}) [TRANSBORDO, peso={ed['weight']}]")
        else:
            print(f"  {nm1} (L{n1[1]}) -> {nm2} (L{n2[1]}) [viaje, peso={ed['weight']}]")
    '''

    # 2. CONVERTIR EN TRAMOS (segmentos)
    tramos = [] # lista final de tramos
    tramo_actual = []  # tramo que estamos construyendo ahora mismo

    for nodo in mejor_camino:
        if not tramo_actual:
            tramo_actual.append(nodo) #primer nodo
        elif tramo_actual[-1][1] == nodo[1]:  
            tramo_actual.append(nodo) #mismas linea?-> continuamos tramo
        else:
            tramos.append(tramo_actual) # cambio de linea -> cerrar tramo actual y guardamos
            tramo_actual = [nodo] # iniciar nuevo tramo con este nodo (transbordo)

    if tramo_actual:
        tramos.append(tramo_actual) #para el ultim tramo que quedo sin cerrar

    if not tramos:
        return []

    # 3. BUSCAR TRENES REALES (GTFS)
    primer_tramo = tramos[0]
    origen_t1 = primer_tramo[0][0] #stop id primera parda
    linea_t1 = primer_tramo[0][1] #lnea

    cantidad_busqueda = 150 if es_llegada else 40
    hora_busqueda = hora_para_comparar if es_llegada else hora_target_min

    trenes_opciones = buscar_trenes_en_gtfs_rapido( #lista trenes candidatos
        linea_t1,
        origen_t1,
        hora_busqueda,
        cantidad=cantidad_busqueda,
        hour=hour_ref,
        dow=dow
    )

    if es_llegada:
        trenes_opciones = sorted(
            trenes_opciones,
            key=lambda x: abs(x['hora_salida_min'] - hora_para_comparar) 
        )

    opciones_finales = []
    horas_usadas = [] #horas_usadas = set() para el problema de que en mta pasan cada al menos 10 min y aqui al superponer registros de diferentes dias, podia dar trenes cada minuto
    max_trenes_evaluar = 60 #max de trenes a evaluar (para no tardar una decad)
    count = 0
    max_delay_historico = int(viajes_df['delay_at_destination'].quantile(0.95))
    for tren_ini in trenes_opciones:

        if count >= max_trenes_evaluar:
            break

        diff = tren_ini['hora_salida_min'] - hora_target_min
        limite_inferior = -90 if es_llegada else -max_delay_historico #IMPORTANTE antes:para lo de que si el usuario termina queriendo salir unos minutos antes, pueda coger un tren anterior a la hora de salida especificada
        #ahora usar el percentil 95 para cubrir el 95 % de todos los retrasos 

        if diff < limite_inferior:
            continue

        #anyadido ultimo
        solapado = False
        for h_usada in horas_usadas:
            if abs(tren_ini['hora_salida_min'] - h_usada) < 6: #media de espacio entre trenes
                solapado = True
                break
        
        if solapado:
            continue

        if tren_ini['hora_salida_min'] in horas_usadas: #si la hora de salida se va a repetir, descartamos ese tren
            continue

        hora_actual = tren_ini['hora_salida_min']
        tiempo_prog_total = 0
        delay_total = 0
        detalles_tramos = []
        detalles_transbordo = []
        viaje_ok = True
        hora_salida_programada_tramo = hora_actual
        for idx, tramo_data in enumerate(tramos):

            ruta_stops = [t[0] for t in tramo_data] #lista de stops_id
            linea_actual = tramo_data[0][1]
            origen_tramo = ruta_stops[0]
            fin_tramo = ruta_stops[-1]

            if idx > 0: #es decir, si no es el primer tramo (primer tocho de segmenteo, lo que quiere decir un transbordo)

                hora_llegada_fisica = hora_actual
                #hora_actual += 3.0 # 3 min de transbordo, preguntar a la profesora si esto le parece bien 

                linea_prev = tramos[idx - 1][0][1]

                detalles_transbordo.append({
                    'estacion': stops_df[stops_df['stop_id'] == origen_tramo]['stop_name'].iloc[0],
                    'de_linea': linea_prev,
                    'a_linea': linea_actual,
                    'tiempo': 3.0
                })

                hora_transbordo_h = int(hora_actual // 60) % 24

                trips_linea = trips_df[trips_df['route_id'] == linea_actual]['trip_id'].unique()
                trips_linea = trips_validos_para_hora(trips_linea, hora_transbordo_h, dow) # de nuevo no usar nocturnos o tal cuando no toca

                stop_times_candidatos = stop_times_df[
                    (stop_times_df['stop_id'].str.replace('[NS]', '', regex=True) == limpiar_stop_id(origen_tramo)) & #paradas q counciden con el punto de transporte
                    (stop_times_df['trip_id'].isin(trips_linea)) # trips validos para la hora en la que este
                ].copy()

                if stop_times_candidatos.empty:
                    viaje_ok = False
                    break

                stop_times_candidatos['hora_min'] = stop_times_candidatos['departure_time'].apply(time_str_to_minutes)

                min_hora_aceptable = hora_llegada_fisica - max_delay_historico

                stop_times_candidatos = stop_times_candidatos[
                    stop_times_candidatos['hora_min'] >= min_hora_aceptable
                ].sort_values('hora_min')

                if stop_times_candidatos.empty:
                    viaje_ok = False
                    break

                best_next = None

                for _, row in stop_times_candidatos.head(50).iterrows():
                    hora_minima = int(row['hora_min'])
                    if hora_minima >= hora_llegada_fisica + TIEMPO_TRANSBORDO - max_delay_historico: #if hora_minima >= hora_actual: #+ TIEMPO_TRANSBORDO:
                        best_next = {
                            'trip_id': row['trip_id'],
                            'hora_salida_min': hora_minima,
                            'hora_salida_prog': row['departure_time']
                        }
                        break
                

                if not best_next:
                    viaje_ok = False
                    break

                hora_salida_programada_tramo = best_next['hora_salida_min']

                # hora real a la que puedes subirte (tras esperar transbordo)
                hora_actual = max(
                    best_next['hora_salida_min'],
                    hora_llegada_fisica + TIEMPO_TRANSBORDO
                )

            # Prediccion del tramo.
            # delay_actual solo se pasa en el primer tramo (tren origen conocido);
            # los tramos de transbordo siempre en modo pre-viaje (tren desconocido).
            res, err = predecir_ruta_completa(
                ruta_stops,
                hora_actual,
                dow,
                grafo=grafo,
                delay_actual=delay_actual if idx == 0 else None
            )

            if err:
                viaje_ok = False
                break

            

            detalles_tramos.append({
                'linea': linea_actual,
                'origen': stops_df[stops_df['stop_id'] == origen_tramo]['stop_name'].iloc[0],
                'destino': stops_df[stops_df['stop_id'] == fin_tramo]['stop_name'].iloc[0],
                'tiempo_prog': res['tiempo_prog'],
                'delay': res['delay_total'],
                'hora_salida_tramo': (
                    int(hora_salida_programada_tramo // 60) % 24,   # ← hora programada real del tren
                    int(hora_salida_programada_tramo % 60)
                )
            })

            tiempo_prog_total += res['tiempo_prog']
            delay_total += res['delay_total']
            hora_actual = res['hora_llegada_min']

        if viaje_ok:

            ruta_nombres = []

            for n in mejor_camino:
                nm = stops_df[stops_df['stop_id'] == n[0]]['stop_name'].iloc[0]
                if not ruta_nombres or ruta_nombres[-1] != nm:
                    ruta_nombres.append(nm)

            tiempo_transbordos = len(detalles_transbordo) * TIEMPO_TRANSBORDO
            tiempo_total_calculado = tiempo_prog_total + tiempo_transbordos

            opciones_finales.append({
                'route_id': linea_t1,
                'camino_str': " -> ".join(ruta_nombres),
                'origen_nom': ruta_nombres[0],
                'destino_nom': ruta_nombres[-1],
                'hora_salida': (
                    int(tren_ini['hora_salida_min'] // 60) % 24,
                    int(tren_ini['hora_salida_min'] % 60)
                ),
                'hora_llegada': (
                    int(hora_actual // 60) % 24,
                    int(hora_actual % 60)
                ),
                'tiempo_programado': tiempo_prog_total,
                'retraso_predicho': delay_total,
                'tiempo_total': tiempo_total_calculado,
                'transbordos': detalles_transbordo,
                'detalles_tramos': detalles_tramos,
                'tramos_stops': [[n[0] for n in tramo] for tramo in tramos],
                'hora_llegada_abs': hora_actual
            })

            horas_usadas.append(tren_ini['hora_salida_min'])
            count += 1

    # 4. ORDENACION 

    for op in opciones_finales:

        hora_lleg_ant = None
        llegada_real = op['hora_llegada_abs']

        for idx, tramo in enumerate(op.get('detalles_tramos', [])):

            hs = tramo.get('hora_salida_tramo')

            if hs:
                sal_est = hs[0] * 60 + hs[1] + tramo['delay']

                if idx > 0 and hora_lleg_ant is not None:
                    sal_est = max(sal_est, hora_lleg_ant + TIEMPO_TRANSBORDO) # para q no pueda salir antes de llegar+trans

                llegada_real = sal_est + tramo['tiempo_prog']
                hora_lleg_ant = llegada_real

        op['hora_llegada_real'] = llegada_real

    if es_llegada:
        opciones_finales.sort(
            key=lambda op: (
                abs(op['hora_llegada_real'] - hora_para_comparar), #primero mas cercano
                op['hora_llegada_real'] - hora_para_comparar # en caso de empate, llegar antes q tarde
            )
        )
        
        opciones_finales = opciones_finales[:3] 
        
    else:
        for op in opciones_finales:
            primer_tramo = op['detalles_tramos'][0]
            hs = primer_tramo['hora_salida_tramo']
            op['hora_salida_estimada_abs'] = hs[0] * 60 + hs[1] + primer_tramo['delay']
            # hora estimada de salida = programada + delay predicho del primer tramo

        opciones_finales.sort(
            key=lambda x: abs(x['hora_salida_estimada_abs'] - hora_target_min)
            # ordenar por cercanía de la salida ESTIMADA a la hora pedida por el usuario
        )
        opciones_finales = opciones_finales[:3]

    return opciones_finales


def calcular_opciones_llegada(origenes, destinos, hora_llegada_str, dow, linea_origen=None, linea_destino=None):

    h_parts = hora_llegada_str.split(':')
    hora_obj_min = int(h_parts[0]) * 60 + int(h_parts[1])
    hour_obj = int(hora_obj_min // 60) % 24

    # Seleccionar grafo segun la hora objetivo
    grafo = seleccionar_grafo(hour_obj, dow)

    ids_origen  = [limpiar_stop_id(o[0]) for o in origenes]
    ids_destino = [limpiar_stop_id(d[0]) for d in destinos]
    nodos_origen  = [n for n in grafo.nodes() if n[0] in ids_origen  and (linea_origen  is None or n[1] == linea_origen)]
    nodos_destino = [n for n in grafo.nodes() if n[0] in ids_destino]

    if not nodos_origen or not nodos_destino:
        print("Error: Los IDs de origen/destino no estan en el grafo.") #(paradas9)
        return []

    # Estimacion rapida del peso para calcular ventana de busqueda
    min_weight = float('inf')
    for start in nodos_origen:
        for end in nodos_destino:
            try:
                w = nx.shortest_path_length(grafo, start, end, weight='weight')
                if w < min_weight:
                    min_weight = w
            except:
                continue

    if min_weight == float('inf'):
        return []

    margen= 10 #por si aca
    #hora_inicio_busqueda = max(0, hora_obj_min - 30 - OVERHEAD_ESTACION - 25)
    hora_inicio_busqueda = max(0, hora_obj_min - min_weight - OVERHEAD_ESTACION - margen) #margen dinamico en funcion del viaje, no es lo mismo 2 parada que 15
    hora_ideal_str = f"{int(hora_inicio_busqueda // 60):02d}:{int(hora_inicio_busqueda % 60):02d}"
    #print(f"DEBUG Buscando trenes desde: {hora_ideal_str} para llegar a {hora_llegada_str}")

    opciones = calcular_opciones_dijkstra(
        origenes, destinos, hora_ideal_str, dow,
        es_llegada=True, hora_objetivo_real=hora_obj_min,
        linea_origen=linea_origen, linea_destino=linea_destino
    )

    # para mostrar al usuario diferencia con tiempo introducido
    for op in opciones:
        llegada_abs = op.get('hora_llegada_abs', op['hora_llegada'][0] * 60 + op['hora_llegada'][1])
        op['diff_objetivo'] = llegada_abs - hora_obj_min
        op['hora_objetivo'] = (int(hora_obj_min // 60), int(hora_obj_min % 60))

    return opciones


def seleccionar_parada(nombre_input, tipo):
    
    candidatos = stops_df[stops_df['stop_name'].str.contains(nombre_input, case=False, na=False)]
    
    # solo la parada tocha( 120, 120N, 120R -> 120)
    # o IDs que no terminen en N o S para el buscador de rutas
    candidatos = candidatos[candidatos['stop_id'].str.match(r'^\d+$|^[A-Z]\d+$')]
    
    if candidatos.empty:
        print(f"No se encontro la estacion '{nombre_input}'")
        return []
        
    #print(f"  DEBUG Buscando '{nombre_input}' ({tipo})...")
    #print(f"  DEBUG Candidatos encontrados: {len(candidatos)}")
    
    return list(zip(candidatos['stop_id'], candidatos['stop_name'])) #devolvemos todo ya que puede existir la misma parada en varias lineas pero
    #con distinto nombre, cogemos todo y ya dejamos q dijjstr eliga



def mostrar_opciones(opciones, modo='salida'):
    if not opciones:
        print("\nNo se encontraron rutas posibles entre las estaciones indicadas.")
        return

    for i, op in enumerate(opciones, 1):

        # --- Calcular horas estimadas recorriendo los tramos ---
        hora_lleg_est_anterior = None
        salida_est_real  = None
        llegada_est_real = None

        for idx, tramo in enumerate(op.get('detalles_tramos', [])):
            hora_salida_tren = tramo.get('hora_salida_tramo')
            if hora_salida_tren:
                salida_prog_min = hora_salida_tren[0] * 60 + hora_salida_tren[1]
                salida_est_min  = salida_prog_min + tramo['delay']

                if idx == 0:
                    salida_est_real = salida_est_min  # guardar salida estimada del primer tramo

                if idx > 0 and hora_lleg_est_anterior is not None:
                    salida_est_min = max(salida_est_min, hora_lleg_est_anterior + TIEMPO_TRANSBORDO)

                llegada_est_min        = salida_est_min + tramo['tiempo_prog']
                hora_lleg_est_anterior = llegada_est_min
                llegada_est_real       = llegada_est_min

        # --- Formatear horas estimadas para la cabecera ---
        if salida_est_real is not None:
            h_salida = f"{int(salida_est_real // 60) % 24:02d}:{int(salida_est_real % 60):02d}"
        else:
            h_salida = f"{op['hora_salida'][0]:02d}:{op['hora_salida'][1]:02d}"

        if llegada_est_real is not None:
            h_llegada = f"{int(llegada_est_real // 60) % 24:02d}:{int(llegada_est_real % 60):02d}"
        else:
            h_llegada = f"{op['hora_llegada'][0]:02d}:{op['hora_llegada'][1]:02d}"

        # --- Imprimir cabecera ---
        print(f"\nOPCION {i} - Linea {op['route_id']}")
        print(f"ORIGEN: {op['origen_nom']} -> DESTINO: {op['destino_nom']}")

        if modo == 'llegada' and 'hora_objetivo' in op:
            # comparar llegada estimada contra la hora objetivo del usuario
            hora_obj_min = op['hora_objetivo'][0] * 60 + op['hora_objetivo'][1]
            diff_real = llegada_est_real - hora_obj_min if llegada_est_real is not None else 0
            if diff_real == 0:
                estado = "A TIEMPO"
            elif diff_real > 0:
                estado = f"+{abs(diff_real):.0f} min TARDE"
            else:
                estado = f"{abs(diff_real):.0f} min ANTES"
            print(f"SALIDA: {h_salida} | LLEGADA: {h_llegada} ({estado})")
            print(f"Objetivo: {op['hora_objetivo'][0]:02d}:{op['hora_objetivo'][1]:02d}")
        else:
            # modo salida: no mostrar estado
            print(f"SALIDA: {h_salida} | LLEGADA: {h_llegada}")

        print(f"\n\tRuta: {op['camino_str']}")
        print("\nDESGLOSE POR TRAMOS:")

        hora_lleg_est_anterior = None  # reset para el desglose

        for idx, tramo in enumerate(op.get('detalles_tramos', [])):

            signo = "+" if tramo['delay'] > 0 else ""

            if idx > 0:
                transbordos = op.get('transbordos', [])
                if idx - 1 < len(transbordos):
                    t = transbordos[idx - 1]
                    print(f"\tTRANSBORDO en {t['estacion']}: Cambiar a L{t['a_linea']} (+{t['tiempo']} min)")

            hora_salida_tren = tramo.get('hora_salida_tramo')
            if hora_salida_tren:
                salida_min     = hora_salida_tren[0] * 60 + hora_salida_tren[1]
                salida_est_min = salida_min + tramo['delay']

                if idx > 0 and hora_lleg_est_anterior is not None:
                    salida_est_min = max(salida_est_min, hora_lleg_est_anterior + TIEMPO_TRANSBORDO)

                llegada_prog_min       = salida_min + tramo['tiempo_prog']
                llegada_est_min        = salida_est_min + tramo['tiempo_prog']
                hora_lleg_est_anterior = llegada_est_min

                h_sal_prog  = f"{hora_salida_tren[0]:02d}:{hora_salida_tren[1]:02d}"
                h_sal_est   = f"{int(salida_est_min // 60) % 24:02d}:{int(salida_est_min % 60):02d}"
                h_lleg_prog = f"{int(llegada_prog_min // 60) % 24:02d}:{int(llegada_prog_min % 60):02d}"
                h_lleg_est  = f"{int(llegada_est_min // 60) % 24:02d}:{int(llegada_est_min % 60):02d}"

                print(f"\t\n Tramo {idx+1} (Linea {tramo['linea']}): {tramo['origen']} -> {tramo['destino']}")
                print(f"\t Salida programada: {h_sal_prog} | Salida estimada: {h_sal_est} (Delay: {signo}{tramo['delay']:.1f} min)")
                print(f"\t Llegada programada: {h_lleg_prog} | Llegada estimada: {h_lleg_est}")
                print(f"\t Tiempo de viaje: {tramo['tiempo_prog']:.1f} min")
            else:
                print(f"Tramo {idx+1} (Linea {tramo['linea']}): {tramo['origen']} -> {tramo['destino']}")

        print("\t", "-"*30)
        print(f"\t= {op['tiempo_total']:.1f} min TOTAL")
        print("-"*80)
        


def modo_salida():
    print("\n" + "-"*40)
    print("MODO: HORA DE SALIDA")
    print("-"*40)

    origen_input  = input("Estacion ORIGEN: ").strip()
    destino_input = input("Estacion DESTINO: ").strip()

    hora_salida = input("Hora de SALIDA (HH:MM) [ENTER para hora actual]: ").strip()

    if not hora_salida:
        hora_salida, dow_actual = obtener_hora_actual()
        fecha_hoy = datetime.now()
        print(f"Usando hora actual: {hora_salida} ({DIAS[dow_actual]} {fecha_hoy.strftime('%d/%m/%Y')})")
        dow = dow_actual
    else:
        fecha_str = input("Fecha de viaje (DD/MM/AAAA) [ENTER para hoy]: ").strip()
        if not fecha_str:
            fecha    = datetime.now()
            dow      = fecha.weekday()
            print(f"Usando fecha de hoy: {fecha.strftime('%d/%m/%Y')} ({DIAS[dow]})")
        else:
            fecha, dow = parsear_fecha(fecha_str)
            if fecha is None:
                print("Formato de fecha incorrecto. Usa DD/MM/AAAA")
                return
            print(f"Fecha seleccionada: {fecha.strftime('%d/%m/%Y')} ({DIAS[dow]})")

    origenes = seleccionar_parada(origen_input, "ORIGEN")
    if not origenes: return

    destinos = seleccionar_parada(destino_input, "DESTINO")
    if not destinos: return

    opciones = calcular_opciones_dijkstra(origenes, destinos, hora_salida, dow)
    mostrar_opciones(opciones, modo='salida')



def modo_llegada():
    print("\n" + "-"*40)
    print("MODO: HORA DE LLEGADA")
    print("-"*40)

    origen_input  = input("Estacion ORIGEN: ").strip()
    destino_input = input("Estacion DESTINO: ").strip()
    hora_llegada  = input("Hora de LLEGADA deseada (HH:MM): ").strip()

    fecha_str = input("Fecha de viaje (DD/MM/AAAA) [ENTER para hoy]: ").strip()
    if not fecha_str:
        fecha = datetime.now()
        dow   = fecha.weekday()
        print(f"Usando fecha de hoy: {fecha.strftime('%d/%m/%Y')} ({DIAS[dow]})")
    else:
        fecha, dow = parsear_fecha(fecha_str)
        if fecha is None:
            print("Formato de fecha incorrecto. Usa DD/MM/AAAA")
            return
        print(f"Fecha seleccionada: {fecha.strftime('%d/%m/%Y')} ({DIAS[dow]})")

    origenes = seleccionar_parada(origen_input, "ORIGEN")
    if not origenes: return

    destinos = seleccionar_parada(destino_input, "DESTINO")
    if not destinos: return

    opciones = calcular_opciones_llegada(origenes, destinos, hora_llegada, dow)
    mostrar_opciones(opciones, modo='llegada')
    

def menu_principal():
    while True:
        print("SELECCIONA MODO:")
        
        print("\n1. Hora de SALIDA (saber cuando llegaras)")
        print("2. Hora de LLEGADA (ver opciones de trenes)")
        
        opcion = input("Opcion (1-2): ").strip()
        
        if opcion == '1':
            modo_salida()
        elif opcion == '2':
            modo_llegada()
        else:
            print("Opcion invalida")


if __name__ == '__main__':
    menu_principal()