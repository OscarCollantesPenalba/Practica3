"""Métricas de calidad del clustering."""
from pyspark.ml.evaluation import ClusteringEvaluator


# Usado en: Ej1
def silhouette(predicciones):
    """Silhouette de las predicciones: cerca de 1 = clusters bien separados, cerca de 0 = solapados."""
    evaluator = ClusteringEvaluator(featuresCol="features",
                                    predictionCol="prediction",
                                    metricName="silhouette")
    return evaluator.evaluate(predicciones)
