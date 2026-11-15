from src.modelo.comun import config
from src.modelo.comun.semillas import fijar_semillas
from src.modelo.comun.spark_session import get_spark_session
from src.vista.ml_view import MLView

# Ejercicios implementados, en orden de ejecución.
EJERCICIOS = []


class MLController:
    """Orquesta los ejercicios: llama a funciones del modelo y pasa los resultados a la vista.
    La lógica vive en src/modelo; aquí solo se encadenan llamadas y se muestran resultados."""

    def __init__(self):
        fijar_semillas(config.SEED)
        self.spark = get_spark_session()
        self.view = MLView()

    def run(self, ejercicios=None):
        for nombre in (ejercicios or EJERCICIOS):
            getattr(self, nombre.lower())()
        self.spark.stop()

#========================================================================================
#===================CLASE 1 - Sistema MLOps (KMeans, Hyperopt, MLflow) ==================
#========================================================================================


#========================================================================================
#===================CLASE 2 - Aprendizaje no supervisado (DBSCAN, GMM) ==================
#========================================================================================


#========================================================================================
#===================CLASE 3 - Análisis y Power BI =======================================
#========================================================================================
