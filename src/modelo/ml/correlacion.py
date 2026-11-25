"""Correlación de Pearson entre variables (análisis previo al clustering)."""
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation


# Usado en: Ej6
def matriz_correlacion(df, columnas):
    """Matriz de correlación de Pearson entre las columnas indicadas (array de numpy).
    Se quitan antes las filas con algún nulo en esas columnas (Sentimiento es nulo en los
    días posteriores al 30/09/2026): VectorAssembler no admite nulos y Pearson necesita
    los valores de todas las variables en cada fila."""
    datos = df.dropna(subset=columnas)
    vec = VectorAssembler(inputCols=columnas, outputCol="features").transform(datos)
    return Correlation.corr(vec, "features").head()[0].toArray()


# Usado en: Ej7
def correlacion(df, columna_a, columna_b):
    """Coeficiente de Pearson entre dos columnas, quitando antes las filas con nulos en ellas."""
    return df.dropna(subset=[columna_a, columna_b]).stat.corr(columna_a, columna_b)
