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

# --- Hiperparámetros ---
VALORES_K_EJ1 = [2, 3, 4, 5, 6, 7, 8]      # k probados a mano en el Ej1

# --- Experimentos de MLflow (uno por algoritmo y dato) ---
EXPERIMENTO_EJ1 = "kmeans_BBVA_manual"     # Ej1: k probados a mano (sin Hyperopt)
EXPERIMENTO_KMEANS_BBVA = "kmeans_BBVA"    # Ej2: búsqueda con Hyperopt
