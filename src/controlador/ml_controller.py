from src.modelo.comun import config
from src.modelo.comun.semillas import fijar_semillas
from src.modelo.comun.spark_session import get_spark_session
from src.modelo.datos import datos_empresa, leer_gold
from src.modelo.ml.tuning import mejor_k, probar_valores_k
from src.modelo.tracking import usar_experimento
from src.vista.ml_view import MLView

# Ejercicios implementados, en orden de ejecución.
EJERCICIOS = ["ej1"]


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

    def ej1(self):
        # Ej1 - Herramienta IA usada: Claude
        self.view.titulo("Ej1")
        # Días de BBVA descritos por su apertura y su cierre.
        hecho = leer_gold(self.spark, config.GOLD_HECHO)
        bbva = datos_empresa(hecho, config.TICKER_BBVA, config.FEATURES_EJ1)
        # Sin Hyperopt: se prueban a mano varios k; cada uno queda como una run en MLflow
        # y se elige el de mayor silhouette.
        usar_experimento(config.EXPERIMENTO_EJ1)
        resultados = probar_valores_k(bbva, config.FEATURES_EJ1, config.VALORES_K_EJ1, config.SEED)
        self.view.valor("Número de agrupaciones", mejor_k(resultados))


#========================================================================================
#===================CLASE 2 - Aprendizaje no supervisado (DBSCAN, GMM) ==================
#========================================================================================


#========================================================================================
#===================CLASE 3 - Análisis y Power BI =======================================
#========================================================================================
