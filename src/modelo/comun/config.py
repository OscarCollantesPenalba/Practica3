"""Configuración central: rutas, semilla, empresa, columnas y nombres de experimentos."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"

# --- Entrada: capa gold de la Práctica 2 (solo lectura) ---
GOLD_DIR = DATA_DIR / "gold"
GOLD_HECHO = (GOLD_DIR / "hecho_cotizacion").as_posix()
GOLD_DIM_EMPRESA = (GOLD_DIR / "dim_empresa").as_posix()
GOLD_DIM_FECHA = (GOLD_DIR / "dim_fecha").as_posix()

# --- Feature store (Delta con columna version) ---
LAKE_DIR = DATA_DIR / "lake"
FEATURE_STORE = (LAKE_DIR / "features").as_posix()

# --- Reproducibilidad ---
SEED = 42

# --- Datos de cada ejercicio ---
TICKER_BBVA = "BBVA.MC"
FEATURES_EJ1 = ["Open", "Close"]
FEATURES_EJ5 = ["Variacion_Diaria_Pct", "Rango_Intradia_Pct"]   # otras columnas de BBVA (Ej5)
VERSION_FEATURES_EJ4 = 1                       # versión del feature store que se guarda y recupera
K_EJ4 = 6                                      # k elegido por Hyperopt en el Ej2

# --- Hiperparámetros ---
VALORES_K_EJ1 = [2, 3, 4, 5, 6, 7, 8, 9, 10]   # k probados a mano en el Ej1 (curva del codo)
K_MIN_EJ2, K_MAX_EJ2 = 2, 10                   # espacio de búsqueda de Hyperopt: k = 2, 3, ..., 10
MAX_EVALS_EJ2 = 10                             # nº de intentos (runs) de Hyperopt

# --- Experimentos de MLflow (uno por algoritmo y dato) ---
EXPERIMENTO_EJ1 = "kmeans_BBVA_manual"     # Ej1: k probados a mano (sin Hyperopt)
EXPERIMENTO_KMEANS_BBVA = "kmeans_BBVA"    # Ej2: búsqueda con Hyperopt
EXPERIMENTO_EJ5 = "kmeans_BBVA_variacion"  # Ej5: misma búsqueda con otras columnas
