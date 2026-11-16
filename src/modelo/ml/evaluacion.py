"""Métricas de calidad del clustering y selección del número de agrupaciones."""
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col


# Usado en: Ej1, Ej2, Ej4
def silhouette(predicciones):
    """Silhouette de las predicciones: cerca de 1 = clusters bien separados, cerca de 0 = solapados."""
    evaluator = ClusteringEvaluator(featuresCol="features",
                                    predictionCol="prediction",
                                    metricName="silhouette")
    return evaluator.evaluate(predicciones)


# Usado en: Ej1
def sse(modelo):
    """Suma de errores cuadráticos (SSE / inercia) de un pipeline con KMeans en la última etapa:
    suma de las distancias al cuadrado de cada punto a su centroide. Siempre baja al aumentar k."""
    return modelo.stages[-1].summary.trainingCost


# Usado en: Ej1
def k_codo(valores_k, valores_sse):
    """Método del codo: k donde la curva (k, SSE) deja de bajar con fuerza.
    Se normalizan ambos ejes a [0, 1] y se elige el punto más alejado de la recta que une
    el primer y el último punto de la curva (el "codo")."""
    k_ini, k_fin = valores_k[0], valores_k[-1]
    sse_ini, sse_fin = valores_sse[0], valores_sse[-1]
    distancias = []
    for k, valor in zip(valores_k, valores_sse):
        x = (k - k_ini) / (k_fin - k_ini)             # 0 en el primer k, 1 en el último
        y = (valor - sse_fin) / (sse_ini - sse_fin)   # 1 en el primer SSE, 0 en el último
        distancias.append((1 - x) - y)                 # distancia a la recta y = 1 - x
    return valores_k[distancias.index(max(distancias))]


# Usado en: Ej4
def dias_con_distinto_cluster(predicciones_a, predicciones_b):
    """Compara dos clasificaciones de los mismos días (clave Ticker, Fecha) y cuenta
    cuántos días han caído en un cluster distinto. 0 = misma clasificación."""
    a = predicciones_a.select("Ticker", "Fecha", col("prediction").alias("cluster_a"))
    b = predicciones_b.select("Ticker", "Fecha", col("prediction").alias("cluster_b"))
    return (a.join(b, on=["Ticker", "Fecha"], how="inner")
             .filter(col("cluster_a") != col("cluster_b"))
             .count())
