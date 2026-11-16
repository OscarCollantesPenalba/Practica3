"""Carga de la capa gold (Práctica 2) y selección de los datos de cada ejercicio."""
from pyspark.sql.functions import col


# Usado en: Ej1, Ej2, Ej4
def leer_gold(spark, ruta):
    """Lee una tabla de la gold (Parquet)."""
    return spark.read.parquet(ruta)


# Usado en: Ej1, Ej2, Ej4
def datos_empresa(hecho, ticker, columnas):
    """Sesiones de una empresa con su clave (Ticker, Fecha) y las columnas pedidas.
    Se quitan las filas con nulos en esas columnas: VectorAssembler falla si encuentra un nulo."""
    return (hecho.filter(col("Ticker") == ticker)
                 .select(["Ticker", "Fecha"] + columnas)
                 .dropna(subset=columnas))
