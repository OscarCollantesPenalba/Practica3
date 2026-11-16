"""Construcción de pipelines de Spark MLlib (Transformers -> Estimator)."""
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler


# Usado en: Ej1, Ej2, Ej4, Ej5
def crear_pipeline_kmeans(columnas, k, seed):
    """Pipeline VectorAssembler (Transformer) -> KMeans (Estimator).
    - VectorAssembler junta las columnas en el vector "features" que necesita MLlib.
    - seed fija la inicialización aleatoria de los centroides: mismo k y mismos datos
      dan siempre los mismos clusters.
    No se añade StandardScaler: apertura y cierre están en la misma unidad (€) y escala."""
    assembler = VectorAssembler(inputCols=columnas, outputCol="features")
    kmeans = KMeans(featuresCol="features", k=k, seed=seed)
    return Pipeline(stages=[assembler, kmeans])
