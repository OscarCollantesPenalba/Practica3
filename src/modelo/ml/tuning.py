"""Búsqueda de hiperparámetros: a mano (Ej1) y con Hyperopt (Ej2)."""
from src.modelo.ml.evaluacion import silhouette, sse
from src.modelo.ml.pipelines import crear_pipeline_kmeans
from src.modelo.tracking import registrar_run


# Usado en: Ej1
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
