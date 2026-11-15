"""Sesión de Spark en modo local con Delta Lake (reutilizada de la Práctica 1)."""
import os
import sys

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from src.modelo.comun.config import BASE_DIR


def get_spark_session(app_name="IBEX35_ML"):
    # Spark lanza procesos Python (workers); se le indica el mismo Python que ejecuta
    # el script (evita el error "Python worker exited unexpectedly" en Windows).
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    builder = (SparkSession.builder
               .appName(app_name)
               .master("local[*]")
               .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog",
                       "org.apache.spark.sql.delta.catalog.DeltaCatalog")
               # Ubicación fija del warehouse dentro del proyecto.
               .config("spark.sql.warehouse.dir",
                       (BASE_DIR / "data" / "spark-warehouse").as_posix()))

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
