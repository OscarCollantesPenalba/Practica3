"""Feature store hecho a mano: guarda las features en Delta con una columna version
y permite recuperar una versión concreta para reentrenar (Ej4)."""
from pathlib import Path

from pyspark.sql.functions import col, lit


# Usado en: Ej4
def leer_version(spark, ruta, version):
    """Recupera las features de una versión concreta del feature store."""
    return (spark.read.format("delta").load(ruta)
                 .filter(col("version") == version))


# Usado en: Ej4
def guardar_version(spark, df, ruta, version):
    """Añade las features (clave Ticker, Fecha + columnas) con su número de versión.
    Si esa versión ya está guardada no se vuelve a escribir: así ejecutar el ejercicio
    varias veces no duplica filas y una versión, una vez guardada, no cambia."""
    if Path(ruta).exists() and leer_version(spark, ruta, version).count() > 0:
        return
    (df.withColumn("version", lit(version))
       .write.format("delta").mode("append").save(ruta))
