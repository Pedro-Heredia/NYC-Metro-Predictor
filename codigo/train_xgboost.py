
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    'is_morning_rush', 'is_lunch_rush', 'is_evening_rush',
    'minute_of_hour', 'minute_of_day',
    'position_ratio', 'is_early_segment', 'is_late_segment',
    'segment_number', 'total_segments',
    'is_line_3', 'is_line_5', 'is_express',
    'hour_x_position', 'rush_x_segment','is_night'
]

# Codificar variables categoricas
encoders = {}
for col in ['origin_stop_id', 'destination_stop_id', 'route_id']:
    if col in viajes_df.columns:
        le = LabelEncoder()
        viajes_df[f'{col}_encoded'] = le.fit_transform(viajes_df[col])
        encoders[col] = le
        available_features.append(f'{col}_encoded')

X = viajes_df[available_features].values
y = viajes_df['delay_at_destination'].values

print(f"Features: {len(available_features)}")
print(f"Samples: {len(X):,}")

# split de datos

print("\n Dividiendo datos...")

# 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

# Escalar datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
    'n_jobs': -1  # Usa todos los cores
}

# Entrenar
model = xgb.XGBRegressor(**params)
model.fit(X_train_scaled, y_train, verbose=False)

train_time = time.time() - start_train

print(f"Modelo entrenado en {train_time:.2f} segundos")

# evaluacion

print("\n Evaluando modelo...")



# Predicciones
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

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

# guardar modelo


# Empaquetar todo lo necesario
model_package = {
    'model': model,
    'scaler': scaler,
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

