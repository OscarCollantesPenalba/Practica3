"""Carga de la capa gold (Práctica 2) y selección de los datos de cada ejercicio."""
from pyspark.sql.functions import avg, col, lag
from pyspark.sql.window import Window


# Usado en: Ej1, Ej2, Ej4, Ej5, Ej6, Ej7, Ej8
def leer_gold(spark, ruta):
    """Lee una tabla de la gold (Parquet)."""
    return spark.read.parquet(ruta)


# Usado en: Ej1, Ej2, Ej4, Ej5
def datos_empresa(hecho, ticker, columnas):
    """Sesiones de una empresa con su clave (Ticker, Fecha) y las columnas pedidas.
    Se quitan las filas con nulos en esas columnas: VectorAssembler falla si encuentra un nulo."""
    return (hecho.filter(col("Ticker") == ticker)
                 .select(["Ticker", "Fecha"] + columnas)
                 .dropna(subset=columnas))


# Usado en: Ej6, Ej7, Ej8
def variables_derivadas(hecho):
    """Añade las variables derivadas de cada fila empresa-día:
    Return = (Close - Open) / Open       -> rentabilidad del día (apertura a cierre)
    RangeLow = (High - Low) / Low         -> amplitud del día respecto al mínimo
    RangeOpen = (High - Low) / Open       -> amplitud del día respecto a la apertura"""
    return (hecho.withColumn("Return", (col("Close") - col("Open")) / col("Open"))
                 .withColumn("RangeLow", (col("High") - col("Low")) / col("Low"))
                 .withColumn("RangeOpen", (col("High") - col("Low")) / col("Open")))


# Usado en: Ej8
def sector_mes(hecho, dim_empresa, dim_fecha):
    """Una fila por sector y mes con la media de todas las filas empresa-día del grupo:
    RetMedioMes = media del Return y SentMedioMes = media del Sentimiento.
    El sector sale de dim_empresa y el año/mes de dim_fecha (modelo en estrella de la gold).
    avg ignora los nulos: los meses sin sentimiento (octubre y noviembre de 2026) quedan
    con SentMedioMes nulo."""
    return (hecho.join(dim_empresa.select("Ticker", "Sector"), on="Ticker", how="inner")
                 .join(dim_fecha.select("Fecha", "Anio", "Mes"), on="Fecha", how="inner")
                 .groupBy("Sector", "Anio", "Mes")
                 .agg(avg("Return").alias("RetMedioMes"),
                      avg("Sentimiento").alias("SentMedioMes")))


# Usado en: Ej8
def sentimiento_mes_anterior(tabla_sector_mes):
    """Añade SentMesAnterior: el SentMedioMes del mes anterior del mismo sector (lag).
    Sirve para ver si el sentimiento de un mes anticipa el retorno del mes siguiente."""
    ventana = Window.partitionBy("Sector").orderBy("Anio", "Mes")
    return tabla_sector_mes.withColumn("SentMesAnterior", lag("SentMedioMes").over(ventana))
