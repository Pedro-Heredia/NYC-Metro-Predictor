import json
from pathlib import Path

# Carga los tipos de parada generados automaticamente desde el GTFS.
# Ejecutar generate_service_types.py para regenerar tras una renovacion de GTFS.
_DATA_FILE = Path(__file__).parent / "service_types_data.json"

_service_types_raw = {}
FULL_TIME_ONLY_ROUTES = set()

if _DATA_FILE.exists():
    with open(_DATA_FILE) as f:
        _data = json.load(f)
    _service_types_raw = _data.get('service_types', {})
    FULL_TIME_ONLY_ROUTES = set(_data.get('full_time_only_routes', []))
else:
    print(f"[service_types] AVISO: {_DATA_FILE} no encontrado. "
          f"Ejecuta generate_service_types.py para generarlo.")

# Convertir claves "route_id|stop_id" a tuplas (route_id, stop_id)
STOP_SERVICE_TYPES = {
    tuple(k.split('|', 1)): v
    for k, v in _service_types_raw.items()
}


def get_stop_service_type(route_id, stop_id):
    return STOP_SERVICE_TYPES.get((route_id, stop_id), 'full_time')


def should_train_stop_here(route_id, stop_id, current_hour, day_of_week):
    service_type = get_stop_service_type(route_id, stop_id)

    if service_type == 'full_time':
        return True

    elif service_type == 'part_time':
        return 6 <= current_hour < 23

    elif service_type == 'rush_hour_only':
        is_weekday = day_of_week < 5
        if not is_weekday:
            return False
        is_morning_rush = 6.5 <= current_hour < 9.5
        is_evening_rush = 15.5 <= current_hour < 20
        return is_morning_rush or is_evening_rush

    elif service_type == 'night_service':
        return current_hour < 6 or current_hour >= 22

    return True


def is_special_service_stop(route_id, stop_id):
    return get_stop_service_type(route_id, stop_id) != 'full_time'


def route_has_special_stops(route_id):
    return route_id not in FULL_TIME_ONLY_ROUTES
