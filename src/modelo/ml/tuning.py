"""Búsqueda de hiperparámetros: a mano (Ej1) y con Hyperopt (Ej2)."""
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from src.modelo.ml.evaluacion import silhouette, sse
from src.modelo.ml.pipelines import crear_pipeline_kmeans
from src.modelo.tracking import registrar_run


# Usado en: Ej1, Ej2
def entrenar_kmeans(datos, columnas, k, seed):
    """Entrena el pipeline con un k concreto y devuelve el modelo y sus predicciones."""
    modelo = crear_pipeline_kmeans(columnas, k, seed).fit(datos)
    return modelo, modelo.transform(datos)


# Usado en: Ej1
def probar_valores_k(datos, columnas, valores_k, seed):
    """Entrena un KMeans por cada k y lo registra como una run de MLflow con su SSE y su silhouette.
    Devuelve la lista de SSE en el mismo orden que valores_k (para el método del codo)."""
    valores_sse = []
    for k in valores_k:
        modelo, predicciones = entrenar_kmeans(datos, columnas, k, seed)
        valor_sse = sse(modelo)
        registrar_run({"k": k, "seed": seed, "features": ",".join(columnas)},
                      {"sse": valor_sse, "silhouette": silhouette(predicciones)}, modelo)
        valores_sse.append(valor_sse)
    return valores_sse


# Usado en: Ej2
def crear_objetivo_kmeans(datos, columnas, seed):
    """Devuelve la función objetivo que Hyperopt llama en cada intento.
    Cada llamada entrena un KMeans con el k propuesto, lo registra como una run de MLflow
    y devuelve -silhouette: Hyperopt siempre MINIMIZA, así el mínimo de -silhouette
    es el máximo de silhouette."""
    def objetivo(params):
        k = int(params["k"])   # quniform devuelve el k como float (6.0)
        modelo, predicciones = entrenar_kmeans(datos, columnas, k, seed)
        valor = silhouette(predicciones)
        registrar_run({"k": k, "seed": seed, "features": ",".join(columnas)},
                      {"silhouette": valor}, modelo)
        return {"loss": -valor, "status": STATUS_OK}
    return objetivo


# Usado en: Ej2
def buscar_k_hyperopt(datos, columnas, k_min, k_max, max_evals, seed):
    """Búsqueda automática del mejor k con Hyperopt (TPE) y devuelve el k elegido."""
    # Espacio de búsqueda: k entero entre k_min y k_max (quniform con paso 1, ordinal).
    espacio = {"k": hp.quniform("k", k_min, k_max, 1)}
    mejor = fmin(fn=crear_objetivo_kmeans(datos, columnas, seed),
                 space=espacio,
                 algo=tpe.suggest,
                 max_evals=max_evals,
                 trials=Trials(),
                 rstate=np.random.default_rng(seed),   # semilla de Hyperopt: qué k se prueban
                 show_progressbar=False)               # sin barra de progreso: "muestra" solo lo pedido
    return int(mejor["k"])
