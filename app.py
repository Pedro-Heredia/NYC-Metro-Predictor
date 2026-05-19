import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from datetime import datetime, date, timedelta
from streamlit_javascript import st_javascript
import predictor as p10
from service_types import (
    get_stop_service_type, should_train_stop_here, STOP_SERVICE_TYPES, FULL_TIME_ONLY_ROUTES
)

#https://python-visualization.github.io/folium/latest/user_guide/map.html
#https://folium.streamlit.app/
#https://python-visualization.github.io/folium/latest/user_guide/plugins/heatmap.html

#python -m streamlit run app.py

st.set_page_config(page_title="NYC Metro Predictor", page_icon="🚇", layout="wide")
st.markdown('<style>.titulo{color:#0066CC;font-size:2.5rem;font-weight:bold;text-align:center}</style>',
            unsafe_allow_html=True)

_hora_navegador = st_javascript("new Date().toLocaleTimeString('es-ES', {hour: '2-digit', minute: '2-digit', hour12: false})")
if _hora_navegador and isinstance(_hora_navegador, str) and ':' in _hora_navegador:
    ahora = datetime.now().replace(hour=int(_hora_navegador.split(':')[0]), minute=int(_hora_navegador.split(':')[1]))
else:
    ahora = datetime.utcnow() + timedelta(hours=2)


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
def construir_datos_heatmap():
    try:
        df = p10.viajes_df.copy()
        if 'destination_stop_id' not in df.columns: return None
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


def diagnosticar_sin_ruta(destino_nombre, linea_destino, hora_h, dow):
    """
    Devuelve (motivo_str, [lineas_alternativas]) si la parada tiene servicio especial
    no activo a esta hora. Devuelve (None, []) si el problema es otro.
    """
    if not linea_destino or linea_destino == 'Todas':
        return None, []

    # Buscar stop_ids que correspondan a destino_nombre en linea_destino
    trips_linea = p10.trips_df[p10.trips_df['route_id'] == linea_destino]['trip_id'].unique()
    if len(trips_linea) == 0:
        return None, []

    stop_ids_raw = p10.stop_times_df[
        p10.stop_times_df['trip_id'].isin(trips_linea)
    ]['stop_id'].unique()

    # Encontrar el stop_id base que tiene ese nombre
    base_encontrado = None
    for sid in stop_ids_raw:
        base = p10.limpiar_stop_id(sid)
        match = p10.stops_df[p10.stops_df['stop_id'] == base]
        if not match.empty and match['stop_name'].iloc[0] == destino_nombre:
            base_encontrado = base
            break

    if base_encontrado is None:
        return None, []

    # Comprobar tipo de servicio (probar variantes N, S y base)
    stype = 'full_time'
    sid_usado = base_encontrado
    for variant in [base_encontrado + 'N', base_encontrado + 'S', base_encontrado]:
        t = get_stop_service_type(linea_destino, variant)
        if t != 'full_time':
            stype = t
            sid_usado = variant
            break

    if stype == 'full_time':
        return None, []

    # Verificar si está activo a esta hora
    if should_train_stop_here(linea_destino, sid_usado, hora_h, dow):
        return None, []  # Está activo, el problema es otro

    tipo_nombres = {
        'night_service':  'servicio nocturno (solo de 00:00 a 06:00h)',
        'part_time':      'servicio parcial (solo de 06:00 a 23:00h)',
        'rush_hour_only': 'servicio solo en hora punta (L-V 06:30-09:30h y 15:30-20:00h)',
    }
    motivo = (f"La parada <b>{destino_nombre}</b> en la Línea {linea_destino} tiene "
              f"{tipo_nombres.get(stype, stype)}, que no está activo a las "
              f"{hora_h:02d}h.")

    # Buscar otras lineas que sirvan esa misma parada con servicio activo ahora
    alternativas = []
    stops_mismo_nombre = p10.stops_df[p10.stops_df['stop_name'] == destino_nombre]
    lineas_revisadas = {linea_destino}

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
            # Verificar activo en esta linea a esta hora
            for v in [base_alt + 'N', base_alt + 'S', base_alt]:
                if should_train_stop_here(la, v, hora_h, dow):
                    alternativas.append(la)
                    break

    return motivo, sorted(set(alternativas))


# ============================================================================
# FUNCIONES DE MAPA
# ============================================================================

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
        hm_data = construir_datos_heatmap()
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
            marcadores.append((lat0, lon0, f"Origen: {tramo['origen']}", "green"))
        else:
            marcadores.append((lat0, lon0, f"Transbordo: {tramo['origen']}", "orange"))
        if idx == len(opcion['detalles_tramos']) - 1:
            marcadores.append((lat1, lon1, f"Destino: {tramo['destino']}", "red"))
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
    'opciones_ruta':      None,
    'origen_nombre':      '',
    'destino_nombre':     '',
    'linea_dest_busqueda': None,
    'hora_h_busqueda':    12,
    'dow_busqueda':       0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================
st.markdown('<div class="titulo"> NYC Metro Predictor</div><br>', unsafe_allow_html=True)
paradas_por_linea = obtener_paradas_por_linea()
LINEAS_OPCIONES = ['Todas'] + sorted(p10.LINEAS_VALIDAS)
DIAS_SEMANA = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

col1, col2 = st.columns(2)
with col1:
    modo = st.radio("MODO:", ["HORA DE SALIDA", "HORA DE LLEGADA"])

    # ORIGEN
    c_orig, c_lo = st.columns([4, 1])
    with c_lo:
        linea_orig_sel = st.selectbox("Línea", LINEAS_OPCIONES, key="linea_orig")
    with c_orig:
        paradas_orig = paradas_por_linea.get(linea_orig_sel, paradas_por_linea['Todas'])
        opciones_orig = ['(seleccionar)'] + paradas_orig
        idx_orig = opciones_orig.index(st.session_state.origen_nombre) \
            if st.session_state.origen_nombre in opciones_orig else 0
        origen_sel = st.selectbox("Estación ORIGEN:", opciones_orig, index=idx_orig, key="sel_origen")
        if origen_sel != '(seleccionar)':
            st.session_state.origen_nombre = origen_sel

    # DESTINO
    c_dest, c_ld = st.columns([4, 1])
    with c_ld:
        linea_dest_sel = st.selectbox("Línea", LINEAS_OPCIONES, key="linea_dest")
    with c_dest:
        paradas_dest = paradas_por_linea.get(linea_dest_sel, paradas_por_linea['Todas'])
        opciones_dest = ['(seleccionar)'] + paradas_dest
        idx_dest = opciones_dest.index(st.session_state.destino_nombre) \
            if st.session_state.destino_nombre in opciones_dest else 0
        destino_sel = st.selectbox("Estación DESTINO:", opciones_dest, index=idx_dest, key="sel_destino")
        if destino_sel != '(seleccionar)':
            st.session_state.destino_nombre = destino_sel

with col2:
    hora_input = st.text_input("Hora (HH:MM):", placeholder=ahora.strftime('%H:%M'))
    fecha_input = st.date_input("Fecha de viaje:", value=date.today())
    dow = fecha_input.weekday()
    st.caption(f" {DIAS_SEMANA[dow]}")
    btn_buscar = st.button(" Ejecutar Búsqueda", use_container_width=True, type="primary")

st.markdown("---")

origen_input = st.session_state.origen_nombre
destino_input = st.session_state.destino_nombre

if btn_buscar:
    if not origen_input or not destino_input:
        st.warning("Selecciona un origen y un destino.")
    else:
        hora_str = hora_input.strip() if hora_input.strip() \
            else f"{ahora.hour:02d}:{ahora.minute:02d}"
        origenes = buscar_estaciones_candidatas(
            origen_input, linea_orig_sel if linea_orig_sel != 'Todas' else None)
        destinos = buscar_estaciones_candidatas(
            destino_input, linea_dest_sel if linea_dest_sel != 'Todas' else None)

        if not origenes or not destinos:
            st.error("No se encontró alguna de las estaciones.")
        else:
            lo = linea_orig_sel if linea_orig_sel != 'Todas' else None
            ld = linea_dest_sel if linea_dest_sel != 'Todas' else None
            hora_h_calc = int(hora_str.split(':')[0])

            # Guardar contexto para el diagnóstico posterior
            st.session_state.linea_dest_busqueda = ld
            st.session_state.hora_h_busqueda     = hora_h_calc
            st.session_state.dow_busqueda        = dow

            with st.spinner("Buscando rutas óptimas..."):
                if "LLEGADA" in modo:
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
        # Diagnóstico de por qué no hay ruta
        motivo, alternativas = diagnosticar_sin_ruta(
            destino_input,
            st.session_state.linea_dest_busqueda,
            st.session_state.hora_h_busqueda,
            st.session_state.dow_busqueda
        )
        if motivo:
            st.markdown(
                f"<div style='background:#fff3cd;border:1px solid #ffc107;border-radius:6px;"
                f"padding:12px 16px;margin-bottom:8px;font-size:15px'> {motivo}</div>",
                unsafe_allow_html=True)
            if alternativas:
                lineas_fmt = ", ".join([f"Línea {la}" for la in alternativas])
                st.markdown(
                    f"<div style='background:#d1ecf1;border:1px solid #bee5eb;border-radius:6px;"
                    f"padding:12px 16px;margin-bottom:8px;font-size:15px'>"
                    f" La parada <b>{destino_input}</b> también tiene servicio en: "
                    f"{lineas_fmt}. Prueba seleccionando una de esas líneas.</div>",
                    unsafe_allow_html=True)
        else:
            st.warning("No se encontraron rutas válidas. Prueba con otra hora o estaciones.")

        if st.button(" Limpiar", key="btn_limpiar_vacio"):
            st.session_state.opciones_ruta = None; st.rerun()

    else:
        col_res, col_map = st.columns([6, 4])
        with col_res:
            st.subheader(" Itinerarios Encontrados")
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
                    if d < 0:   diff_txt, diff_class = f"({abs(d)} min ANTES)", "res-early"
                    elif d > 0: diff_txt, diff_class = f"(+{d} min TARDE)", "res-late"
                    else:       diff_txt, diff_class = "(A TIEMPO)", "res-ontime"

                html_output += f"<div class='res-box'>"
                html_output += f"<p class='res-title'>OPCION {i} — línea {op['route_id']}</p>"
                html_output += (f"<p class='res-header'>ORIGEN: {op['detalles_tramos'][0]['origen']}"
                                f" → DESTINO: {op['detalles_tramos'][-1]['destino']}</p>")
                html_output += (f"<p class='res-header'>SALIDA: {h_salida} | LLEGADA: {h_llegada}"
                                f" <span class='{diff_class}'>{diff_txt}</span></p>")
                if 'hora_objetivo' in op:
                    html_output += (f"<p class='res-detail'>Objetivo: "
                                    f"{op['hora_objetivo'][0]:02d}:{op['hora_objetivo'][1]:02d}</p>")
                html_output += f"<p class='res-sep'>────────────────────────────────</p>"
                html_output += f"<p class='res-ruta'><b>Ruta:</b> {op.get('camino_str', '')}</p>"
                html_output += f"<p class='res-sep'>────────────────────────────────</p>"
                html_output += f"<p class='res-header'>&nbsp;&nbsp;DESGLOSE POR TRAMOS:</p>"

                hora_lleg_ant = None
                for idx_t, tramo in enumerate(op['detalles_tramos'], 1):
                    if idx_t > 1:
                        transbordos = op.get('transbordos', [])
                        if idx_t - 2 < len(transbordos):
                            t = transbordos[idx_t - 2]
                            html_output += (f"<p class='res-transbordo'>&nbsp;&nbsp;&nbsp;"
                                            f" TRANSBORDO en {t['estacion']}: "
                                            f"Cambiar a L{t['a_linea']} (+{t['tiempo']} min)</p>")
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
                    html_output += (f"<p class='res-tramo'>Tramo {idx_t} (L{tramo['linea']}): "
                                    f"{tramo['origen']} → {tramo['destino']}</p>")
                    html_output += (f"<p class='res-detail'>Salida prog: {h_sp} | est: {h_se}"
                                    f" <span style='color:#888'>({signo}{delay:.1f} min)</span></p>")
                    html_output += f"<p class='res-detail'>Llegada prog: {h_lp} | est: {h_le}</p>"
                    html_output += f"<p class='res-detail'>Tiempo de viaje: {tramo['tiempo_prog']:.1f} min</p>"
                html_output += (f"<p class='res-sep'>&nbsp;&nbsp;──────────────</p>"
                                f"<p class='res-total'>&nbsp;&nbsp;= {op['tiempo_total']:.1f} min TOTAL</p>"
                                f"</div>")

            st.markdown(html_output, unsafe_allow_html=True)
            if st.button(" Limpiar búsqueda", key="btn_limpiar"):
                st.session_state.opciones_ruta = None; st.rerun()

        with col_map:
            st.subheader(" Visualización")
            st_folium(crear_mapa_ruta_especifica(opciones[0]),
                      width="100%", height=700, key="map_res")

else:
    # ---- MAPA GENERAL ----
    st.subheader(" Mapa Interactivo de la Red Actual")
    col_ctrl, col_mapa = st.columns([1, 4])
    with col_ctrl:
        st.markdown("**Filtrar líneas:**")
        lineas_activas = set()
        for lid in sorted(p10.LINEAS_VALIDAS):
            if st.checkbox(f"Línea {lid}", value=True, key=f"ck_{lid}"):
                lineas_activas.add(lid)
        if not lineas_activas:
            lineas_activas = set(p10.LINEAS_VALIDAS)
        st.markdown("---")
        mostrar_hm = st.checkbox("Retrasos actuales", value=False)

    with col_mapa:
        
        m_general = crear_mapa_general(lineas_activas, mostrar_hm, ahora.hour, ahora.weekday())
        st_folium(m_general, width="100%", height=650, key="map_main")

        # Reloj de hora actual (referencia para el heatmap)
        st.caption(f"Hora actual: **{ahora.strftime('%H:%M')}** - {DIAS_SEMANA[ahora.weekday()]}"
                   f" ({ahora.strftime('%d/%m/%Y')})"
        )
        if mostrar_hm:
            st.markdown(
                "<div style='font-size:13px;line-height:1.8;margin-top:2px'>"
                "<b>Leyenda heatmap:</b><br>"
                " <b>Rojo/Naranja</b> -> retrasos altos (&gt;6 min de media)<br>"
                " <b>Amarillo/Verde</b> -> retrasos moderados (3-6 min)<br>"
                " <b>Azul</b> -> retrasos bajos (&lt;3 min)"
                "</div>",
                unsafe_allow_html=True
            )
        
#https://python-visualization.github.io/folium/latest/user_guide/map.html
#https://folium.streamlit.app/
#https://python-visualization.github.io/folium/latest/user_guide/plugins/heatmap.html

#python -m streamlit run app.py
