import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from datetime import datetime, date, timedelta, timezone
from streamlit_javascript import st_javascript
import predictor as p10
from service_types import (
    get_stop_service_type, should_train_stop_here, STOP_SERVICE_TYPES, FULL_TIME_ONLY_ROUTES
)

try:
    from zoneinfo import ZoneInfo as _ZoneInfo
    _NYC_TZ = _ZoneInfo('America/New_York')
except Exception:
    _NYC_TZ = None

#https://python-visualization.github.io/folium/latest/user_guide/map.html
#https://folium.streamlit.app/
#https://python-visualization.github.io/folium/latest/user_guide/plugins/heatmap.html

#python -m streamlit run app.py

# ============================================================================
# TRADUCCIONES (EN por defecto, cambiable a ES/DE)
# ============================================================================
LANG_LABELS = {'en': '🇬🇧 EN', 'es': '🇪🇸 ES', 'de': '🇩🇪 DE'}

T = {
    'en': {
        'mode_label': 'MODE:', 'mode_departure': 'DEPARTURE TIME', 'mode_arrival': 'ARRIVAL TIME',
        'line_label': 'Line', 'origin_label': 'ORIGIN Station:', 'destination_label': 'DESTINATION Station:',
        'select_placeholder': '(select)', 'all_lines': 'All',
        'time_label': 'Time (HH:MM):', 'date_label': 'Travel date:', 'run_search': 'Run Search',
        'warn_select_both': 'Select an origin and a destination.',
        'err_station_not_found': "One of the stations wasn't found.",
        'err_invalid_time': 'Invalid time format. Use HH:MM (e.g. 17:30).',
        'spinner_searching': 'Searching for optimal routes...',
        'origin_tag': 'ORIGIN', 'destination_tag': 'DESTINATION',
        'warn_no_routes': 'No valid routes found. Try another time or stations.',
        'btn_clear': 'Clear', 'btn_clear_search': 'Clear search',
        'itineraries_found': 'Itineraries Found', 'visualization': 'Visualization',
        'option_word': 'OPTION', 'line_word': 'line',
        'origin_word': 'ORIGIN', 'destination_word': 'DESTINATION',
        'departure_word': 'DEPARTURE', 'arrival_word': 'ARRIVAL', 'target_word': 'Target',
        'route_word': 'Route', 'breakdown_word': 'SEGMENT BREAKDOWN',
        'transfer_word': 'TRANSFER at', 'change_to': 'Switch to L',
        'segment_word': 'Segment', 'sched_word': 'sched', 'est_word': 'est',
        'departure_short': 'Departure', 'arrival_short': 'Arrival',
        'travel_time_word': 'Travel time', 'total_word': 'TOTAL',
        'min_early': 'min EARLY', 'min_late': 'min LATE', 'on_time': 'ON TIME',
        'map_title': 'Interactive Map of the Current Network', 'filter_lines': 'Filter lines:',
        'btn_all': 'All', 'btn_none': 'None', 'current_delays': 'Current delays',
        'current_time_label': 'Current time',
        'heatmap_legend_title': 'Heatmap legend:',
        'heatmap_high': 'Red/Orange -> high delays (&gt;6 min avg)',
        'heatmap_med': 'Yellow/Green -> moderate delays (3-6 min)',
        'heatmap_low': 'Blue -> low delays (&lt;3 min)',
        'about_model': 'ℹ️ About the model',
        'last_retrain': 'Last retraining', 'train_samples': 'Training samples',
        'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        'svc_night_service': 'night service only (22:00 to 06:00)',
        'svc_part_time': 'part-time service only (06:00 to 23:00)',
        'svc_rush_hour_only': 'rush-hour-only service (Mon-Fri 06:30-09:30 and 15:30-20:00)',
        'restriction_msg': 'Station <b>{name}</b> on Line {line} has {svc}, which is not active at {hour:02d}h.',
        'alt_lines_msg': 'Station <b>{name}</b> also has service on: {lines}. Try selecting one of those lines.',
    },
    'es': {
        'mode_label': 'MODO:', 'mode_departure': 'HORA DE SALIDA', 'mode_arrival': 'HORA DE LLEGADA',
        'line_label': 'Línea', 'origin_label': 'Estación ORIGEN:', 'destination_label': 'Estación DESTINO:',
        'select_placeholder': '(seleccionar)', 'all_lines': 'Todas',
        'time_label': 'Hora (HH:MM):', 'date_label': 'Fecha de viaje:', 'run_search': 'Ejecutar Búsqueda',
        'warn_select_both': 'Selecciona un origen y un destino.',
        'err_station_not_found': 'No se encontró alguna de las estaciones.',
        'err_invalid_time': 'Formato de hora inválido. Usa HH:MM (ejemplo: 17:30).',
        'spinner_searching': 'Buscando rutas óptimas...',
        'origin_tag': 'ORIGEN', 'destination_tag': 'DESTINO',
        'warn_no_routes': 'No se encontraron rutas válidas. Prueba con otra hora o estaciones.',
        'btn_clear': 'Limpiar', 'btn_clear_search': 'Limpiar búsqueda',
        'itineraries_found': 'Itinerarios Encontrados', 'visualization': 'Visualización',
        'option_word': 'OPCION', 'line_word': 'línea',
        'origin_word': 'ORIGEN', 'destination_word': 'DESTINO',
        'departure_word': 'SALIDA', 'arrival_word': 'LLEGADA', 'target_word': 'Objetivo',
        'route_word': 'Ruta', 'breakdown_word': 'DESGLOSE POR TRAMOS',
        'transfer_word': 'TRANSBORDO en', 'change_to': 'Cambiar a L',
        'segment_word': 'Tramo', 'sched_word': 'prog', 'est_word': 'est',
        'departure_short': 'Salida', 'arrival_short': 'Llegada',
        'travel_time_word': 'Tiempo de viaje', 'total_word': 'TOTAL',
        'min_early': 'min ANTES', 'min_late': 'min TARDE', 'on_time': 'A TIEMPO',
        'map_title': 'Mapa Interactivo de la Red Actual', 'filter_lines': 'Filtrar líneas:',
        'btn_all': 'Todas', 'btn_none': 'Ninguna', 'current_delays': 'Retrasos actuales',
        'current_time_label': 'Hora actual',
        'heatmap_legend_title': 'Leyenda heatmap:',
        'heatmap_high': 'Rojo/Naranja -> retrasos altos (&gt;6 min de media)',
        'heatmap_med': 'Amarillo/Verde -> retrasos moderados (3-6 min)',
        'heatmap_low': 'Azul -> retrasos bajos (&lt;3 min)',
        'about_model': 'ℹ️ Sobre el modelo',
        'last_retrain': 'Último reentrenamiento', 'train_samples': 'Muestras de entrenamiento',
        'days': ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'],
        'svc_night_service': 'servicio nocturno (solo de 22:00 a 06:00h)',
        'svc_part_time': 'servicio parcial (solo de 06:00 a 23:00h)',
        'svc_rush_hour_only': 'servicio solo en hora punta (L-V 06:30-09:30h y 15:30-20:00h)',
        'restriction_msg': 'La parada <b>{name}</b> en la Línea {line} tiene {svc}, que no está activo a las {hour:02d}h.',
        'alt_lines_msg': 'La parada <b>{name}</b> también tiene servicio en: {lines}. Prueba seleccionando una de esas líneas.',
    },
    'de': {
        'mode_label': 'MODUS:', 'mode_departure': 'ABFAHRTSZEIT', 'mode_arrival': 'ANKUNFTSZEIT',
        'line_label': 'Linie', 'origin_label': 'Start-Station:', 'destination_label': 'Ziel-Station:',
        'select_placeholder': '(auswählen)', 'all_lines': 'Alle',
        'time_label': 'Uhrzeit (HH:MM):', 'date_label': 'Reisedatum:', 'run_search': 'Suche starten',
        'warn_select_both': 'Wähle einen Start und ein Ziel aus.',
        'err_station_not_found': 'Eine der Stationen wurde nicht gefunden.',
        'err_invalid_time': 'Ungültiges Zeitformat. Verwende HH:MM (z. B. 17:30).',
        'spinner_searching': 'Suche nach optimalen Routen...',
        'origin_tag': 'START', 'destination_tag': 'ZIEL',
        'warn_no_routes': 'Keine gültigen Routen gefunden. Versuche eine andere Zeit oder Stationen.',
        'btn_clear': 'Zurücksetzen', 'btn_clear_search': 'Suche zurücksetzen',
        'itineraries_found': 'Gefundene Routen', 'visualization': 'Visualisierung',
        'option_word': 'OPTION', 'line_word': 'Linie',
        'origin_word': 'START', 'destination_word': 'ZIEL',
        'departure_word': 'ABFAHRT', 'arrival_word': 'ANKUNFT', 'target_word': 'Zielzeit',
        'route_word': 'Route', 'breakdown_word': 'STRECKENÜBERSICHT',
        'transfer_word': 'UMSTIEG in', 'change_to': 'Wechseln zu L',
        'segment_word': 'Abschnitt', 'sched_word': 'geplant', 'est_word': 'gesch',
        'departure_short': 'Abfahrt', 'arrival_short': 'Ankunft',
        'travel_time_word': 'Fahrzeit', 'total_word': 'GESAMT',
        'min_early': 'Min früher', 'min_late': 'Min später', 'on_time': 'PÜNKTLICH',
        'map_title': 'Interaktive Karte des aktuellen Netzes', 'filter_lines': 'Linien filtern:',
        'btn_all': 'Alle', 'btn_none': 'Keine', 'current_delays': 'Aktuelle Verspätungen',
        'current_time_label': 'Aktuelle Zeit',
        'heatmap_legend_title': 'Heatmap-Legende:',
        'heatmap_high': 'Rot/Orange -> hohe Verspätungen (&gt;6 Min. im Schnitt)',
        'heatmap_med': 'Gelb/Grün -> mittlere Verspätungen (3-6 Min.)',
        'heatmap_low': 'Blau -> geringe Verspätungen (&lt;3 Min.)',
        'about_model': 'ℹ️ Über das Modell',
        'last_retrain': 'Letztes Training', 'train_samples': 'Trainingsdaten',
        'days': ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag'],
        'svc_night_service': 'nur Nachtbetrieb (22:00 bis 06:00 Uhr)',
        'svc_part_time': 'nur Teilzeitbetrieb (06:00 bis 23:00 Uhr)',
        'svc_rush_hour_only': 'nur Stoßzeitenbetrieb (Mo-Fr 06:30-09:30 und 15:30-20:00 Uhr)',
        'restriction_msg': 'Die Station <b>{name}</b> der Linie {line} hat {svc}, was um {hour:02d} Uhr nicht aktiv ist.',
        'alt_lines_msg': 'Die Station <b>{name}</b> wird auch bedient von: {lines}. Versuche eine dieser Linien.',
    },
}


def t(key, **kwargs):
    s = T[st.session_state.lang].get(key, T['en'].get(key, key))
    return s.format(**kwargs) if kwargs else s


st.set_page_config(page_title="NYC Metro Predictor", page_icon="🚇", layout="wide")
st.markdown("""
<style>
div.block-container{padding-top:0rem !important}
/* Titulo-boton: centrar a base de flex en cada nivel posible del wrapper,
   en vez de solo en el propio <button>, porque text-align no reposiciona
   el elemento dentro de su padre, solo el texto dentro de si mismo. */
div.st-key-header_titulo,
div.st-key-header_titulo > div,
div.st-key-header_titulo [data-testid="stButton"]{
    width:100% !important;display:flex !important;justify-content:center !important;
}
div.st-key-header_titulo button{
    background:none !important;border:none !important;
    padding:0 !important;margin:0 0 1rem 0 !important;cursor:pointer !important;
}
div.st-key-header_titulo button p,
div.st-key-header_titulo button div{
    color:#0066CC !important;font-size:2.5rem !important;font-weight:bold !important;
}
div.st-key-header_titulo button:hover p{color:#0055aa !important;text-decoration:underline !important}
div.st-key-header_titulo button:focus{box-shadow:none !important;outline:none !important}

/* Ocultar la caja del componente st_javascript (solo se usa para leer la
   hora del navegador, no deberia verse) sin afectar a otros iframes (mapa) */
div.st-key-hora_navegador_wrap{height:0 !important;overflow:hidden !important;margin:0 !important}
div.st-key-hora_navegador_wrap iframe{height:0 !important;border:none !important}
</style>
""", unsafe_allow_html=True)

if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

with st.container(key="hora_navegador_wrap"):
    _hora_navegador = st_javascript("new Date().toLocaleTimeString('es-ES', {hour: '2-digit', minute: '2-digit', hour12: false})")
if _hora_navegador and isinstance(_hora_navegador, str) and ':' in _hora_navegador:
    ahora = datetime.now().replace(hour=int(_hora_navegador.split(':')[0]), minute=int(_hora_navegador.split(':')[1]))
else:
    if _NYC_TZ:
        ahora = datetime.now(_NYC_TZ).replace(tzinfo=None)
    else:
        ahora = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=4)


@st.cache_data
def construir_datos_mapa():
    colores = {}
    try:
        r_df = pd.read_csv('routes.txt')
        for _, row in r_df.iterrows():
            colores[str(row['route_id']).strip()] = f"#{row['route_color']}"
    except: pass

    segmentos, nodos = {}, {}
    for r_id in p10.LINEAS_VALIDAS:
        viajes_ruta = p10.trips_df[p10.trips_df['route_id'] == r_id]
        if viajes_ruta.empty: continue
        for dir_id in [0, 1]:
            viajes_dir = viajes_ruta[viajes_ruta['direction_id'] == dir_id]
            if viajes_dir.empty: continue
            mejor_trip, max_p = None, 0
            for tid in viajes_dir['trip_id'].tolist()[:30]:
                n = len(p10.stop_times_df[p10.stop_times_df['trip_id'] == tid])
                if n > max_p: max_p = n; mejor_trip = tid
            if mejor_trip is None: continue
            st_seq = p10.stop_times_df[p10.stop_times_df['trip_id'] == mejor_trip].sort_values('stop_sequence')
            coords_seq = []
            for sid in st_seq['stop_id']:
                base_id = p10.limpiar_stop_id(sid)
                match = p10.stops_df[p10.stops_df['stop_id'].str.startswith(base_id)]
                if not match.empty:
                    lat, lon = match['stop_lat'].iloc[0], match['stop_lon'].iloc[0]
                    nombre = match['stop_name'].iloc[0]
                    coords_seq.append((lat, lon))
                    if (lat, lon) not in nodos:
                        nodos[(lat, lon)] = {'name': nombre, 'lines': set()}
                    nodos[(lat, lon)]['lines'].add(r_id)
            for i in range(len(coords_seq) - 1):
                c1, c2 = coords_seq[i], coords_seq[i+1]
                if c1 == c2: continue
                seg = tuple(sorted([c1, c2]))
                if seg not in segmentos: segmentos[seg] = set()
                segmentos[seg].add(r_id)
    return segmentos, nodos, colores


@st.cache_data
def construir_datos_heatmap(lineas_activas=None):
    try:
        df = p10.viajes_df.copy()
        if 'destination_stop_id' not in df.columns: return None
        if lineas_activas is not None and 'route_id' in df.columns:
            df = df[df['route_id'].astype(str).isin(lineas_activas)]
            if df.empty: return None
        df['stop_base'] = df['destination_stop_id'].astype(str).apply(p10.limpiar_stop_id)
        agrupado = df.groupby(['stop_base','hour','day_of_week'])['delay_at_destination'].mean().reset_index()
        agrupado.columns = ['stop_base','hour','dow','mean_delay']
        sc = p10.stops_df[p10.stops_df['stop_id'].str.match(r'^\d+$|^[A-Z]\d+$')][['stop_id','stop_lat','stop_lon']].copy()
        sc.columns = ['stop_base','lat','lon']
        return agrupado.merge(sc, on='stop_base', how='left').dropna(subset=['lat','lon'])
    except Exception as e:
        print(f"[heatmap] {e}"); return None


@st.cache_data
def obtener_paradas_por_linea():
    result, all_names = {}, set()
    for r_id in sorted(p10.LINEAS_VALIDAS):
        trips = p10.trips_df[p10.trips_df['route_id'] == r_id]['trip_id'].unique()
        if len(trips) == 0: continue
        st_ruta = p10.stop_times_df[p10.stop_times_df['trip_id'].isin(trips)]
        stop_ids = st_ruta['stop_id'].apply(p10.limpiar_stop_id).unique()
        paradas = set()
        for sid in stop_ids:
            match = p10.stops_df[p10.stops_df['stop_id'] == sid]
            if not match.empty: paradas.add(match['stop_name'].iloc[0])
        result[r_id] = sorted(paradas)
        all_names.update(paradas)
    result['Todas'] = sorted(all_names)
    return result


def buscar_estaciones_candidatas(texto, linea=None):
    texto_lower = str(texto).strip().lower()
    df = p10.stops_df
    stop_ids_linea = None
    if linea and linea != 'Todas':
        trips_linea = p10.trips_df[p10.trips_df['route_id'] == linea]['trip_id'].unique()
        stop_ids_linea = set(
            p10.stop_times_df[p10.stop_times_df['trip_id'].isin(trips_linea)]['stop_id']
            .apply(p10.limpiar_stop_id)
        )
    exactos = df[df['stop_name'].str.lower() == texto_lower]
    if stop_ids_linea is not None:
        exactos = exactos[exactos['stop_id'].apply(p10.limpiar_stop_id).isin(stop_ids_linea)]
    if not exactos.empty:
        candidatos, ids_vistos = [], set()
        for _, row in exactos.iterrows():
            limpio = p10.limpiar_stop_id(row['stop_id'])
            if limpio not in ids_vistos:
                ids_vistos.add(limpio); candidatos.append((limpio, row['stop_name']))
        return candidatos
    matches = df[df['stop_name'].str.lower().str.contains(texto_lower, na=False)]
    if stop_ids_linea is not None:
        matches = matches[matches['stop_id'].apply(p10.limpiar_stop_id).isin(stop_ids_linea)]
    candidatos, ids_vistos = [], set()
    for _, row in matches.iterrows():
        limpio = p10.limpiar_stop_id(row['stop_id'])
        if limpio not in ids_vistos:
            ids_vistos.add(limpio); candidatos.append((limpio, row['stop_name']))
    return candidatos


def diagnosticar_restriccion_parada(stop_nombre, linea, hora_h, dow):
    """
    Comprueba si una parada tiene restricción de servicio no activa a esa hora.
    Devuelve (motivo_str, [lineas_alternativas]) si hay restricción,
    o (None, []) si no hay o no se puede determinar.
    """
    if not linea or linea == 'Todas':
        return None, []

    trips_linea = p10.trips_df[p10.trips_df['route_id'] == linea]['trip_id'].unique()
    if len(trips_linea) == 0:
        return None, []

    stop_ids_raw = p10.stop_times_df[
        p10.stop_times_df['trip_id'].isin(trips_linea)
    ]['stop_id'].unique()

    base_encontrado = None
    for sid in stop_ids_raw:
        base = p10.limpiar_stop_id(sid)
        match = p10.stops_df[p10.stops_df['stop_id'] == base]
        if not match.empty and match['stop_name'].iloc[0] == stop_nombre:
            base_encontrado = base
            break

    if base_encontrado is None:
        return None, []

    stype = 'full_time'
    sid_usado = base_encontrado
    for variant in [base_encontrado + 'N', base_encontrado + 'S', base_encontrado]:
        ty = get_stop_service_type(linea, variant)
        if ty != 'full_time':
            stype = ty
            sid_usado = variant
            break

    if stype == 'full_time':
        return None, []

    if should_train_stop_here(linea, sid_usado, hora_h, dow):
        return None, []

    svc_txt = t(f'svc_{stype}') if f'svc_{stype}' in T[st.session_state.lang] else stype
    motivo = t('restriction_msg', name=stop_nombre, line=linea, svc=svc_txt, hour=hora_h)

    alternativas = []
    stops_mismo_nombre = p10.stops_df[p10.stops_df['stop_name'] == stop_nombre]
    lineas_revisadas = {linea}

    for _, row in stops_mismo_nombre.iterrows():
        base_alt = p10.limpiar_stop_id(row['stop_id'])
        trips_alt = p10.stop_times_df[
            p10.stop_times_df['stop_id'].str.replace('[NS]$', '', regex=True) == base_alt
        ]['trip_id'].unique()
        for tid in trips_alt[:20]:
            la_series = p10.trips_df[p10.trips_df['trip_id'] == tid]['route_id']
            if la_series.empty: continue
            la = str(la_series.iloc[0])
            if la in lineas_revisadas or la not in p10.LINEAS_VALIDAS: continue
            lineas_revisadas.add(la)
            for v in [base_alt + 'N', base_alt + 'S', base_alt]:
                if should_train_stop_here(la, v, hora_h, dow):
                    alternativas.append(la)
                    break

    return motivo, sorted(set(alternativas))


def diagnosticar_sin_ruta(destino_nombre, linea_destino, hora_h, dow):
    return diagnosticar_restriccion_parada(destino_nombre, linea_destino, hora_h, dow)


# ============================================================================
# FUNCIONES DE MAPA
# ============================================================================

@st.cache_data
def crear_mapa_general(lineas_activas, mostrar_hm=False, hora_hm=0, dow_hm=0):
    segmentos, nodos, colores = construir_datos_mapa()
    m = folium.Map(location=[40.758, -73.9855], zoom_start=12,
                   tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", attr="Google Maps")
    for (c1, c2), lineas in segmentos.items():
        visibles = lineas & lineas_activas
        if not visibles: continue
        color = colores.get(sorted(visibles)[0], "#555555")
        folium.PolyLine(locations=[c1, c2], color=color, weight=4, opacity=0.8,
                        tooltip="Líneas: " + ", ".join(sorted(visibles))).add_to(m)
    for (lat, lon), info in nodos.items():
        visibles = info['lines'] & lineas_activas
        if not visibles: continue
        folium.CircleMarker(
            location=[lat, lon], radius=5, color="#000000", weight=1.5,
            fill=True, fillColor="#FFFFFF", fillOpacity=1,
            tooltip=f"{info['name']} (lineas: {', '.join(sorted(visibles))})"
        ).add_to(m)
    if mostrar_hm:
        hm_data = construir_datos_heatmap(lineas_activas)
        if hm_data is not None:
            filtrado = hm_data[
                (hm_data['dow'] == dow_hm) &
                (hm_data['hour'].between(max(0, hora_hm-1), min(23, hora_hm+1)))
            ]
            if filtrado.empty:
                filtrado = hm_data[
                    (hm_data['dow'] == dow_hm) &
                    (hm_data['hour'].between(max(0, hora_hm-3), min(23, hora_hm+3)))
                ]
            if filtrado.empty: filtrado = hm_data[hm_data['dow'] == dow_hm]
            if filtrado.empty: filtrado = hm_data
            if not filtrado.empty:
                max_d = filtrado['mean_delay'].clip(lower=0).max()
                if max_d > 0:
                    HeatMap([[r['lat'], r['lon'], max(0, r['mean_delay']) / max_d]
                             for _, r in filtrado.iterrows()],
                            radius=22, blur=15, max_zoom=13).add_to(m)
    return m


def crear_mapa_ruta_especifica(opcion):
    m = folium.Map(tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", attr="Google Maps")
    colores = {}
    try:
        r_df = pd.read_csv('routes.txt')
        for _, row in r_df.iterrows():
            colores[str(row['route_id']).strip()] = f"#{row['route_color']}"
    except: pass
    todas_coords, marcadores = [], []
    tramos_stops = opcion.get('tramos_stops')
    for idx, tramo in enumerate(opcion['detalles_tramos']):
        color_tramo = colores.get(str(tramo['linea']), "#0055cc")
        coords_tramo, stops_tramo = [], []
        if tramos_stops and idx < len(tramos_stops):
            for sid in tramos_stops[idx]:
                match = p10.stops_df[p10.stops_df['stop_id'] == sid]
                if not match.empty:
                    lat, lon, nom = match['stop_lat'].iloc[0], match['stop_lon'].iloc[0], match['stop_name'].iloc[0]
                    coords_tramo.append((lat, lon)); stops_tramo.append((lat, lon, nom))
        else:
            for nombre in [tramo['origen'], tramo['destino']]:
                match = p10.stops_df[p10.stops_df['stop_name'] == nombre]
                if not match.empty:
                    lat, lon = match['stop_lat'].iloc[0], match['stop_lon'].iloc[0]
                    coords_tramo.append((lat, lon)); stops_tramo.append((lat, lon, nombre))
        if not coords_tramo: continue
        folium.PolyLine(locations=coords_tramo, color=color_tramo, weight=6, opacity=0.85,
                        tooltip=f"Línea {tramo['linea']}").add_to(m)
        todas_coords.extend(coords_tramo)
        lat0, lon0 = coords_tramo[0]; lat1, lon1 = coords_tramo[-1]
        if idx == 0:
            marcadores.append((lat0, lon0, f"{t('origin_tag')}: {tramo['origen']}", "green"))
        else:
            marcadores.append((lat0, lon0, f"{t('transfer_word')} {tramo['origen']}", "orange"))
        if idx == len(opcion['detalles_tramos']) - 1:
            marcadores.append((lat1, lon1, f"{t('destination_tag')}: {tramo['destino']}", "red"))
        for i, (lat, lon, nom) in enumerate(stops_tramo):
            if i != 0 and i != len(stops_tramo) - 1:
                folium.CircleMarker(location=[lat, lon], radius=4, color=color_tramo, weight=2,
                                    fill=True, fillColor="#FFFFFF", fillOpacity=0.9, tooltip=nom).add_to(m)
    if todas_coords: m.fit_bounds(todas_coords)
    for lat, lon, texto, color in marcadores:
        folium.Marker(location=[lat, lon], tooltip=texto,
                      icon=folium.Icon(color=color, icon="info-sign")).add_to(m)
    return m



# ============================================================================
# SESSION STATE
# ============================================================================
defaults = {
    'opciones_ruta':       None,
    'origen_nombre':       '',
    'destino_nombre':      '',
    'linea_orig_busqueda': None,
    'linea_dest_busqueda': None,
    'hora_h_busqueda':     12,
    'dow_busqueda':        0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================
_sp_lang, col_lang = st.columns([6, 1])
with col_lang:
    _langs = ['en', 'es', 'de']
    lang_sel = st.selectbox(
        "🌐", options=_langs, format_func=lambda k: LANG_LABELS[k],
        index=_langs.index(st.session_state.lang),
        key="lang_selector", label_visibility="collapsed"
    )
    st.session_state.lang = lang_sel

with st.container(key="header_titulo"):
    if st.button("🚇 NYC Metro Predictor", key="btn_titulo_home"):
        st.session_state.opciones_ruta = None
        st.rerun()

paradas_por_linea = obtener_paradas_por_linea()
LINEAS_OPCIONES = ['Todas'] + sorted(p10.LINEAS_VALIDAS)
DIAS_SEMANA = t('days')

col1, col2 = st.columns(2)
with col1:
    modo = st.radio(
        t('mode_label'), ['salida', 'llegada'],
        format_func=lambda x: t('mode_departure') if x == 'salida' else t('mode_arrival')
    )

    # ORIGEN
    c_orig, c_lo = st.columns([4, 1])
    with c_lo:
        linea_orig_sel = st.selectbox(
            t('line_label'), LINEAS_OPCIONES, key="linea_orig",
            format_func=lambda x: t('all_lines') if x == 'Todas' else x
        )
    with c_orig:
        paradas_orig = paradas_por_linea.get(linea_orig_sel, paradas_por_linea['Todas'])
        opciones_orig = ['(seleccionar)'] + paradas_orig
        idx_orig = opciones_orig.index(st.session_state.origen_nombre) \
            if st.session_state.origen_nombre in opciones_orig else 0
        origen_sel = st.selectbox(
            t('origin_label'), opciones_orig, index=idx_orig, key="sel_origen",
            format_func=lambda x: t('select_placeholder') if x == '(seleccionar)' else x
        )
        if origen_sel != '(seleccionar)':
            st.session_state.origen_nombre = origen_sel

    # DESTINO
    c_dest, c_ld = st.columns([4, 1])
    with c_ld:
        linea_dest_sel = st.selectbox(
            t('line_label'), LINEAS_OPCIONES, key="linea_dest",
            format_func=lambda x: t('all_lines') if x == 'Todas' else x
        )
    with c_dest:
        paradas_dest = paradas_por_linea.get(linea_dest_sel, paradas_por_linea['Todas'])
        opciones_dest = ['(seleccionar)'] + paradas_dest
        idx_dest = opciones_dest.index(st.session_state.destino_nombre) \
            if st.session_state.destino_nombre in opciones_dest else 0
        destino_sel = st.selectbox(
            t('destination_label'), opciones_dest, index=idx_dest, key="sel_destino",
            format_func=lambda x: t('select_placeholder') if x == '(seleccionar)' else x
        )
        if destino_sel != '(seleccionar)':
            st.session_state.destino_nombre = destino_sel

with col2:
    hora_input = st.text_input(t('time_label'), placeholder=ahora.strftime('%H:%M'))
    fecha_input = st.date_input(t('date_label'), value=date.today())
    dow = fecha_input.weekday()
    st.caption(f" {DIAS_SEMANA[dow]}")
    btn_buscar = st.button(f" {t('run_search')}", use_container_width=True, type="primary")

st.markdown("---")

origen_input = st.session_state.origen_nombre
destino_input = st.session_state.destino_nombre

if btn_buscar:
    if not origen_input or not destino_input:
        st.warning(t('warn_select_both'))
    else:
        hora_str = hora_input.strip() if hora_input.strip() \
            else f"{ahora.hour:02d}:{ahora.minute:02d}"
        origenes = buscar_estaciones_candidatas(
            origen_input, linea_orig_sel if linea_orig_sel != 'Todas' else None)
        destinos = buscar_estaciones_candidatas(
            destino_input, linea_dest_sel if linea_dest_sel != 'Todas' else None)

        if not origenes or not destinos:
            st.error(t('err_station_not_found'))
        else:
            lo = linea_orig_sel if linea_orig_sel != 'Todas' else None
            ld = linea_dest_sel if linea_dest_sel != 'Todas' else None
            try:
                partes = hora_str.split(':')
                hora_h_calc = int(partes[0])
                if not (0 <= hora_h_calc <= 23) or len(partes) < 2 or not (0 <= int(partes[1]) <= 59):
                    raise ValueError
            except (ValueError, IndexError):
                st.error(t('err_invalid_time'))
                st.stop()

            # Guardar contexto para el diagnóstico posterior
            st.session_state.linea_orig_busqueda = lo
            st.session_state.linea_dest_busqueda = ld
            st.session_state.hora_h_busqueda     = hora_h_calc
            st.session_state.dow_busqueda        = dow

            with st.spinner(t('spinner_searching')):
                if modo == 'llegada':
                    st.session_state.opciones_ruta = p10.calcular_opciones_llegada(
                        origenes=origenes, destinos=destinos,
                        hora_llegada_str=hora_str, dow=dow,
                        linea_origen=lo, linea_destino=ld)
                else:
                    st.session_state.opciones_ruta = p10.calcular_opciones_dijkstra(
                        origenes=origenes, destinos=destinos,
                        hora_salida_prog=hora_str, dow=dow, es_llegada=False,
                        linea_origen=lo, linea_destino=ld)


# ============================================================================
# RENDERIZADO: RESULTADOS O MAPA GENERAL
# ============================================================================
if st.session_state.opciones_ruta is not None:
    opciones = st.session_state.opciones_ruta

    if len(opciones) == 0:
        hora_h_dx  = st.session_state.hora_h_busqueda
        dow_dx     = st.session_state.dow_busqueda

        motivo_orig, alt_orig = diagnosticar_restriccion_parada(
            origen_input,
            st.session_state.linea_orig_busqueda,
            hora_h_dx, dow_dx
        )
        motivo_dest, alt_dest = diagnosticar_restriccion_parada(
            destino_input,
            st.session_state.linea_dest_busqueda,
            hora_h_dx, dow_dx
        )

        mostrado = False
        for motivo, alternativas, stop_label, stop_nombre in [
            (motivo_orig, alt_orig, t('origin_tag'), origen_input),
            (motivo_dest, alt_dest, t('destination_tag'), destino_input),
        ]:
            if motivo:
                mostrado = True
                st.markdown(
                    f"<div style='background:#fff3cd;border:1px solid #ffc107;border-radius:6px;"
                    f"padding:12px 16px;margin-bottom:8px;font-size:15px'>"
                    f"<b>[{stop_label}]</b> {motivo}</div>",
                    unsafe_allow_html=True)
                if alternativas:
                    lineas_fmt = ", ".join([f"{t('line_label')} {la}" for la in alternativas])
                    st.markdown(
                        f"<div style='background:#d1ecf1;border:1px solid #bee5eb;border-radius:6px;"
                        f"padding:12px 16px;margin-bottom:8px;font-size:15px'>"
                        f"{t('alt_lines_msg', name=stop_nombre, lines=lineas_fmt)}</div>",
                        unsafe_allow_html=True)

        if not mostrado:
            st.warning(t('warn_no_routes'))

        if st.button(f" {t('btn_clear')}", key="btn_limpiar_vacio"):
            st.session_state.opciones_ruta = None; st.rerun()

    else:
        col_res, col_map = st.columns([5.68, 4.32])  # mapa +8%, resultados ajustado en consecuencia
        with col_res:
            st.subheader(f" {t('itineraries_found')}")
            html_output = """<style>
            .res-box{font-family:'Calibri','Segoe UI',sans-serif;font-size:16px;line-height:1.6;
              background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;
              padding:12px 16px;margin-bottom:12px}
            .res-title{font-size:17px;font-weight:bold;color:#0066cc;margin:0 0 3px 0}
            .res-header{font-weight:bold;margin:3px 0}
            .res-ruta{color:#333;margin:4px 0;word-break:break-word}
            .res-sep{color:#bbb;margin:4px 0}
            .res-tramo{margin:4px 0 0 12px;font-weight:bold}
            .res-detail{margin:1px 0 0 24px;color:#555;font-size:15px}
            .res-transbordo{margin:5px 0 5px 12px;color:#c07000;font-weight:bold}
            .res-total{margin:4px 0 0 12px;font-weight:bold;font-size:17px}
            .res-late{color:#cc0000}.res-early{color:#007700}.res-ontime{color:#0055aa}
            </style>"""

            for i, op in enumerate(opciones, 1):
                hora_lleg_ant = None; salida_est_real = None; llegada_est_real = None
                for idx_t, tramo in enumerate(op.get('detalles_tramos', [])):
                    hs = tramo.get('hora_salida_tramo')
                    if hs:
                        sal_est = hs[0] * 60 + hs[1] + tramo['delay']
                        if idx_t == 0: salida_est_real = sal_est
                        if idx_t > 0 and hora_lleg_ant is not None:
                            sal_est = max(sal_est, hora_lleg_ant + p10.TIEMPO_TRANSBORDO)
                        lleg_est = sal_est + tramo['tiempo_prog']
                        hora_lleg_ant = lleg_est; llegada_est_real = lleg_est

                h_salida = (f"{int(salida_est_real // 60) % 24:02d}:{int(salida_est_real % 60):02d}"
                            if salida_est_real is not None
                            else f"{op['hora_salida'][0]:02d}:{op['hora_salida'][1]:02d}")
                h_llegada = (f"{int(llegada_est_real // 60) % 24:02d}:{int(llegada_est_real % 60):02d}"
                             if llegada_est_real is not None
                             else f"{op['hora_llegada'][0]:02d}:{op['hora_llegada'][1]:02d}")

                diff_txt = ""; diff_class = ""
                if 'hora_objetivo' in op and llegada_est_real is not None:
                    hora_obj_min = op['hora_objetivo'][0] * 60 + op['hora_objetivo'][1]
                    d = int(llegada_est_real - hora_obj_min)
                    if d < -720: d += 1440
                    if d < 0:   diff_txt, diff_class = f"({abs(d)} {t('min_early')})", "res-early"
                    elif d > 0: diff_txt, diff_class = f"(+{d} {t('min_late')})", "res-late"
                    else:       diff_txt, diff_class = f"({t('on_time')})", "res-ontime"

                html_output += f"<div class='res-box'>"
                html_output += f"<p class='res-title'>{t('option_word')} {i} — {t('line_word')} {op['route_id']}</p>"
                html_output += (f"<p class='res-header'>{t('origin_word')}: {op['detalles_tramos'][0]['origen']}"
                                f" → {t('destination_word')}: {op['detalles_tramos'][-1]['destino']}</p>")
                html_output += (f"<p class='res-header'>{t('departure_word')}: {h_salida} | {t('arrival_word')}: {h_llegada}"
                                f" <span class='{diff_class}'>{diff_txt}</span></p>")
                if 'hora_objetivo' in op:
                    html_output += (f"<p class='res-detail'>{t('target_word')}: "
                                    f"{op['hora_objetivo'][0]:02d}:{op['hora_objetivo'][1]:02d}</p>")
                html_output += f"<p class='res-sep'>────────────────────────────────</p>"
                html_output += f"<p class='res-ruta'><b>{t('route_word')}:</b> {op.get('camino_str', '')}</p>"
                html_output += f"<p class='res-sep'>────────────────────────────────</p>"
                html_output += f"<p class='res-header'>&nbsp;&nbsp;{t('breakdown_word')}:</p>"

                hora_lleg_ant = None
                for idx_t, tramo in enumerate(op['detalles_tramos'], 1):
                    if idx_t > 1:
                        transbordos = op.get('transbordos', [])
                        if idx_t - 2 < len(transbordos):
                            tb = transbordos[idx_t - 2]
                            html_output += (f"<p class='res-transbordo'>&nbsp;&nbsp;&nbsp;"
                                            f" {t('transfer_word')} {tb['estacion']}: "
                                            f"{t('change_to')}{tb['a_linea']} (+{tb['tiempo']} min)</p>")
                    hs = tramo['hora_salida_tramo']
                    delay = tramo.get('delay', 0.0); signo = "+" if delay > 0 else ""
                    sal_est = hs[0] * 60 + hs[1] + delay
                    if idx_t > 1 and hora_lleg_ant is not None:
                        sal_est = max(sal_est, hora_lleg_ant + p10.TIEMPO_TRANSBORDO)
                    lleg_prog = hs[0] * 60 + hs[1] + tramo['tiempo_prog']
                    lleg_est = sal_est + tramo['tiempo_prog']; hora_lleg_ant = lleg_est
                    h_sp = f"{hs[0]:02d}:{hs[1]:02d}"
                    h_se = f"{int(sal_est // 60) % 24:02d}:{int(sal_est % 60):02d}"
                    h_lp = f"{int(lleg_prog // 60) % 24:02d}:{int(lleg_prog % 60):02d}"
                    h_le = f"{int(lleg_est // 60) % 24:02d}:{int(lleg_est % 60):02d}"
                    html_output += (f"<p class='res-tramo'>{t('segment_word')} {idx_t} (L{tramo['linea']}): "
                                    f"{tramo['origen']} → {tramo['destino']}</p>")
                    html_output += (f"<p class='res-detail'>{t('departure_short')} {t('sched_word')}: {h_sp} | {t('est_word')}: {h_se}"
                                    f" <span style='color:#888'>({signo}{delay:.1f} min)</span></p>")
                    html_output += f"<p class='res-detail'>{t('arrival_short')} {t('sched_word')}: {h_lp} | {t('est_word')}: {h_le}</p>"
                    html_output += f"<p class='res-detail'>{t('travel_time_word')}: {tramo['tiempo_prog']:.1f} min</p>"
                html_output += (f"<p class='res-sep'>&nbsp;&nbsp;──────────────</p>"
                                f"<p class='res-total'>&nbsp;&nbsp;= {op['tiempo_total']:.1f} min {t('total_word')}</p>"
                                f"</div>")

            st.markdown(html_output, unsafe_allow_html=True)
            if st.button(f" {t('btn_clear_search')}", key="btn_limpiar"):
                st.session_state.opciones_ruta = None; st.rerun()

        with col_map:
            st.subheader(f" {t('visualization')}")
            st_folium(crear_mapa_ruta_especifica(opciones[0]),
                      width="100%", height=700, key="map_res")

else:
    # ---- MAPA GENERAL ----
    st.subheader(f" {t('map_title')}")
    col_ctrl, col_mapa = st.columns([1, 4])
    with col_ctrl:
        st.markdown(f"**{t('filter_lines')}**")
        lineas_ordenadas = sorted(p10.LINEAS_VALIDAS)

        b_all, b_none = st.columns(2)
        with b_all:
            if st.button(t('btn_all'), use_container_width=True, key="btn_lineas_todas"):
                st.session_state.pills_lineas = lineas_ordenadas
        with b_none:
            if st.button(t('btn_none'), use_container_width=True, key="btn_lineas_ninguna"):
                st.session_state.pills_lineas = []

        st.markdown("""
        <style>
        div.st-key-wrap_pills_lineas [data-testid="stPills"] > div{
            flex-wrap:wrap !important; overflow-x:visible !important; height:auto !important;
        }
        </style>
        """, unsafe_allow_html=True)
        with st.container(key="wrap_pills_lineas"):
            seleccion = st.pills(
                "Líneas a mostrar en el mapa", lineas_ordenadas, default=lineas_ordenadas,
                selection_mode="multi", key="pills_lineas", label_visibility="collapsed"
            )
        lineas_activas = frozenset(seleccion)
        st.markdown("---")
        mostrar_hm = st.checkbox(t('current_delays'), value=False)

    with col_mapa:

        m_general = crear_mapa_general(lineas_activas, mostrar_hm, ahora.hour, ahora.weekday())
        st_folium(m_general, width="100%", height=650, key="map_main")

        # Reloj de hora actual (referencia para el heatmap)
        st.caption(f"{t('current_time_label')}: **{ahora.strftime('%H:%M')}** - {DIAS_SEMANA[ahora.weekday()]}"
                   f" ({ahora.strftime('%d/%m/%Y')})"
        )
        if mostrar_hm:
            st.markdown(
                "<div style='font-size:13px;line-height:1.8;margin-top:2px'>"
                f"<b>{t('heatmap_legend_title')}</b><br>"
                f" {t('heatmap_high')}<br>"
                f" {t('heatmap_med')}<br>"
                f" {t('heatmap_low')}"
                "</div>",
                unsafe_allow_html=True
            )

st.markdown("---")
with st.expander(t('about_model')):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (test)", f"{p10.mae_test:.2f} min" if p10.mae_test is not None else "—")
    c2.metric("R² (test)", f"{p10.r2_test*100:.1f}%" if p10.r2_test is not None else "—")
    c3.metric("RMSE (test)", f"{p10.rmse_test:.2f} min" if p10.rmse_test is not None else "—")
    c4.metric("MAPE (test)", f"{p10.mape_test:.1f}%" if p10.mape_test is not None else "—")
    detalles = []
    if p10.train_date:
        detalles.append(f"**{t('last_retrain')}:** {p10.train_date}")
    if p10.n_samples_train:
        detalles.append(f"**{t('train_samples')}:** {p10.n_samples_train:,}")
    if detalles:
        st.caption(" · ".join(detalles))

#https://python-visualization.github.io/folium/latest/user_guide/map.html
#https://folium.streamlit.app/
#https://python-visualization.github.io/folium/latest/user_guide/plugins/heatmap.html

#python -m streamlit run app.py
