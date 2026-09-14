
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import pickle
import time
import warnings
warnings.filterwarnings('ignore')


# carga de datos


start_total = time.time()

viajes_df = pd.read_pickle('dataset_viajes_raw.pkl')
print(f"Dataset cargado: {len(viajes_df):,} registros")

# preparar features

print("\n Preparando features...")

# Features disponibles ANTES del viaje (sin data leakage)
available_features = [
    'hour', 'day_of_week', 'travel_time_minutes',
    'is_morning_rush', 'is_evening_rush',
    'minute_of_hour', 'minute_of_day',
    'position_ratio', 'is_early_segment', 'is_late_segment',
    'segment_number', 'total_segments',
    'is_line_3', 'is_line_5', 'is_express',
    'hour_x_position', 'rush_x_segment', 'is_night',
    'origin_is_part_time', 'dest_is_part_time',
    'origin_is_rush_hour', 'dest_is_rush_hour',
    'delay_at_origin', 'cumulative_delay_origin',
    'month', 'is_weekend', 'direction_north'
]

# Codificar variables categoricas
encoders = {}
for col in ['origin_stop_id', 'destination_stop_id', 'route_id']:
    if col in viajes_df.columns:
        le = LabelEncoder()
        viajes_df[f'{col}_encoded'] = le.fit_transform(viajes_df[col])
        encoders[col] = le
        available_features.append(f'{col}_encoded')

print(f"Features: {len(available_features)}")
print(f"Samples: {len(viajes_df):,}")

# split temporal de datos

print("\n Dividiendo datos (split temporal)...")

# Ordenar por timestamp y usar 80% mas antiguo como train, 20% mas reciente como test.
# Evita que el modelo vea el futuro durante el entrenamiento (data leakage temporal).
viajes_df = viajes_df.sort_values('timestamp').reset_index(drop=True)
cutoff = viajes_df['timestamp'].quantile(0.8)
train_df = viajes_df[viajes_df['timestamp'] <= cutoff]
test_df  = viajes_df[viajes_df['timestamp'] > cutoff]

X_train = train_df[available_features].values.astype(float)
y_train = train_df['delay_at_destination'].values
X_test  = test_df[available_features].values.astype(float)
y_test  = test_df['delay_at_destination'].values

# Simular consultas pre-viaje: enmascarar delay_at_origin y cumulative_delay_origin
# en el 40% del train. XGBoost aprende a funcionar en ambos modos:
#   - con valores reales  -> modo tiempo real (tren ya en camino, delay conocido)
#   - con NaN             -> modo pre-viaje   (consulta futura, delay desconocido)
delay_origin_idx = available_features.index('delay_at_origin')
cumul_origin_idx = available_features.index('cumulative_delay_origin')
rng = np.random.default_rng(42)
pretrain_mask = rng.random(len(X_train)) < 0.4
X_train[pretrain_mask, delay_origin_idx] = np.nan
X_train[pretrain_mask, cumul_origin_idx] = np.nan
print(f"Muestras modo pre-viaje (NaN): {pretrain_mask.sum():,} ({pretrain_mask.mean()*100:.0f}%)")

print(f"Cutoff: {cutoff}")
print(f"Train: {len(X_train):,} ({len(X_train)/len(viajes_df)*100:.1f}%)")
print(f"Test:  {len(X_test):,} ({len(X_test)/len(viajes_df)*100:.1f}%)")

# entrenamiento

print("\nEntrenando XGBoost...")


start_train = time.time()

# Hiperparametros optimizados (del archivo 6)
params = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'random_state': 42,
    'n_jobs': -1
}

# Entrenar
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train, verbose=False)

train_time = time.time() - start_train

print(f"Modelo entrenado en {train_time:.2f} segundos")

# evaluacion

print("\n Evaluando modelo...")



# Predicciones
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metricas Train
mae_train  = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train   = r2_score(y_train, y_pred_train)
# MAPE: excluir muestras con y==0 para evitar division por cero
mask_train = y_train != 0
mape_train = mean_absolute_percentage_error(y_train[mask_train], y_pred_train[mask_train]) * 100

# Metricas Test
mae_test  = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test   = r2_score(y_test, y_pred_test)
mask_test = y_test != 0
mape_test = mean_absolute_percentage_error(y_test[mask_test], y_pred_test[mask_test]) * 100

print(f"{'Metrica':<15} {'Train':<20} {'Test':<20}")
print("-" * 70)
print(f"{'MAE':<15} {mae_train:.3f} min ({mae_train*60:.0f} seg)  {mae_test:.3f} min ({mae_test*60:.0f} seg)")
print(f"{'RMSE':<15} {rmse_train:.3f} min ({rmse_train*60:.0f} seg)  {rmse_test:.3f} min ({rmse_test*60:.0f} seg)")
print(f"{'R2':<15} {r2_train:.4f} ({r2_train*100:.1f}%)     {r2_test:.4f} ({r2_test*100:.1f}%)")
print(f"{'MAPE':<15} {mape_train:.1f}%                {mape_test:.1f}%")
print("-" * 70)

# Analisis de errores
errors_test = y_pred_test - y_test

within_30s = (np.abs(errors_test) <= 0.5).sum() / len(errors_test) * 100
within_60s = (np.abs(errors_test) <= 1.0).sum() / len(errors_test) * 100
within_120s = (np.abs(errors_test) <= 2.0).sum() / len(errors_test) * 100

print(f"\nPrecision por rangos (Test):")
print(f"  +-30 seg:  {within_30s:>5.1f}%")
print(f"  +-60 seg:  {within_60s:>5.1f}%")
print(f"  +-120 seg: {within_120s:>5.1f}%")

# Evaluacion por modo: tiempo real vs pre-viaje
X_test_rt = X_test.copy()
X_test_pt = X_test.copy()
X_test_pt[:, delay_origin_idx] = np.nan
X_test_pt[:, cumul_origin_idx] = np.nan

y_pred_rt = model.predict(X_test_rt)
y_pred_pt = model.predict(X_test_pt)

mae_rt = mean_absolute_error(y_test, y_pred_rt)
mae_pt = mean_absolute_error(y_test, y_pred_pt)
r2_rt  = r2_score(y_test, y_pred_rt)
r2_pt  = r2_score(y_test, y_pred_pt)

print(f"\nEvaluacion por modo:")
print(f"  Tiempo real (delay_at_origin conocido): MAE={mae_rt:.3f} min  R2={r2_rt:.4f} ({r2_rt*100:.1f}%)")
print(f"  Pre-viaje   (delay_at_origin = NaN):    MAE={mae_pt:.3f} min  R2={r2_pt:.4f} ({r2_pt*100:.1f}%)")

# guardar modelo


# Empaquetar todo lo necesario
model_package = {
    'model': model,
    'encoders': encoders,
    'features': available_features,
    'mae_test': mae_test,
    'rmse_test': rmse_test,
    'r2_test': r2_test,
    'mape_test': mape_test,
    'params': params,
    'train_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test)
}

filename = 'modelo_xgboost_final.pkl'
with open(filename, 'wb') as f:
    pickle.dump(model_package, f)

print(f"Modelo guardado en: {filename}")


file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
print(f"Tamano del archivo: {file_size:.2f} MB")

# resumen

total_time = time.time() - start_total



print(f"\nRendimiento del modelo:")
print(f"  MAE Test:  {mae_test:.3f} min ({mae_test*60:.0f} seg)")
print(f"   RMSE Test: {rmse_test:.3f} min ({rmse_test*60:.0f} seg)")
print(f"   R2 Test:   {r2_test:.4f} ({r2_test*100:.1f}%)")
print(f"   MAPE Test: {mape_test:.1f}%")

print(f"\nPrecision:")
print(f"   {within_60s:.1f}% de predicciones dentro de +-60 segundos")

print(f"\nTiempos:")
print(f"   Entrenamiento: {train_time:.2f} seg")
print(f"   Total:         {total_time:.2f} seg")

print(f"\nArchivo generado:")
print(f"   {filename} ({file_size:.2f} MB)")

print(f"\nModelo listo para usar en la aplicacion!")
