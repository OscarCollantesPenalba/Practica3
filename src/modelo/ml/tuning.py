"""Búsqueda de hiperparámetros: a mano (Ej1) y con Hyperopt (Ej2)."""
from src.modelo.ml.evaluacion import silhouette
from src.modelo.ml.pipelines import crear_pipeline_kmeans
from src.modelo.tracking import registrar_run


# Usado en: Ej1
def entrenar_kmeans(datos, columnas, k, seed):
    """Entrena el pipeline con un k concreto y devuelve el modelo y sus predicciones."""
    modelo = crear_pipeline_kmeans(columnas, k, seed).fit(datos)
    return modelo, modelo.transform(datos)


# Usado en: Ej1
def probar_valores_k(datos, columnas, valores_k, seed):
    """Entrena un KMeans por cada k, lo registra como una run de MLflow y devuelve [(k, silhouette)]."""
    resultados = []
    for k in valores_k:
        modelo, predicciones = entrenar_kmeans(datos, columnas, k, seed)
        valor = silhouette(predicciones)
        registrar_run({"k": k, "seed": seed, "features": ",".join(columnas)},
                      {"silhouette": valor}, modelo)
        resultados.append((k, valor))
    return resultados


# Usado en: Ej1
def mejor_k(resultados):
    """k con la mayor silhouette."""
    return max(resultados, key=lambda r: r[1])[0]
