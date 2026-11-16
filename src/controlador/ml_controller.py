from src.modelo.comun import config
from src.modelo.comun.semillas import fijar_semillas
from src.modelo.comun.spark_session import get_spark_session
from src.modelo.datos import datos_empresa, leer_gold
from src.modelo.ml.evaluacion import k_codo
from src.modelo.ml.tuning import probar_valores_k
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
        # Sin Hyperopt: se entrena un KMeans por cada k y cada uno queda como una run en MLflow
        # (con su SSE y su silhouette, para compararlos en la interfaz).
        # El número de agrupaciones se elige con el método del codo: el k a partir del cual
        # añadir más clusters apenas reduce el SSE.
        usar_experimento(config.EXPERIMENTO_EJ1)
        valores_sse = probar_valores_k(bbva, config.FEATURES_EJ1, config.VALORES_K_EJ1, config.SEED)
        self.view.valor("Número de agrupaciones", k_codo(config.VALORES_K_EJ1, valores_sse))


#========================================================================================
#===================CLASE 2 - Aprendizaje no supervisado (DBSCAN, GMM) ==================
#========================================================================================


#========================================================================================
#===================CLASE 3 - Análisis y Power BI =======================================
#========================================================================================
