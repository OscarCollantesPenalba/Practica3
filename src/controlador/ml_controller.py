from src.modelo.comun import config
from src.modelo.comun.semillas import fijar_semillas
from src.modelo.comun.spark_session import get_spark_session
from src.modelo.datos import datos_empresa, leer_gold, variables_derivadas
from src.modelo.feature_store import guardar_version, leer_version
from src.modelo.ml.correlacion import correlacion, matriz_correlacion
from src.modelo.ml.evaluacion import dias_con_distinto_cluster, k_codo, silhouette
from src.modelo.ml.tuning import buscar_k_hyperopt, entrenar_kmeans, probar_valores_k
from src.modelo.tracking import mejores_runs, runs_ordenadas, usar_experimento
from src.vista.ml_view import MLView

# Ejercicios implementados, en orden de ejecución.
EJERCICIOS = ["ej1", "ej2", "ej3", "ej4", "ej5", "ej6", "ej7"]
EJERCICIOS = ["ej7"]

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

    def ej2(self):
        # Ej2 - Herramienta IA usada: Claude
        self.view.titulo("Ej2")
        hecho = leer_gold(self.spark, config.GOLD_HECHO)
        bbva = datos_empresa(hecho, config.TICKER_BBVA, config.FEATURES_EJ1)
        # Hyperopt prueba k entre 2 y 10 (MAX_EVALS_EJ2 intentos) eligiendo con TPE los más
        # prometedores; cada intento queda como una run en el experimento kmeans_BBVA.
        # Métrica: silhouette (distinta del SSE del Ej1). Se devuelve -silhouette porque Hyperopt minimiza.
        #
        # Las dos semillas (config.SEED):
        # - Seed del modelo, KMeans(seed=...): fija la posición inicial aleatoria de los centroides.
        #   Con el mismo k y los mismos datos, cada entrenamiento da siempre los mismos clusters
        #   y la misma silhouette.
        # - Seed de Hyperopt, fmin(rstate=np.random.default_rng(...)): fija qué valores de k
        #   propone la búsqueda y en qué orden. Sin ella, cada ejecución probaría otros k.
        # Con las dos fijas, toda la búsqueda se repite igual: mismos k probados, mismas
        # métricas y mismo k elegido.
        usar_experimento(config.EXPERIMENTO_KMEANS_BBVA)
        k = buscar_k_hyperopt(bbva, config.FEATURES_EJ1, config.K_MIN_EJ2, config.K_MAX_EJ2,
                              config.MAX_EVALS_EJ2, config.SEED)
        self.view.valor("Número de agrupaciones seleccionado por Hyperopt", k)

    def ej3(self):
        # Ej3 - Herramienta IA usada: Claude
        self.view.titulo("Ej3")
        # Todas las runs del experimento del Ej2 (kmeans_BBVA), de mayor a menor silhouette.
        # Requiere haber ejecutado antes el Ej2 (si no, el experimento no existe).
        runs = runs_ordenadas(config.EXPERIMENTO_KMEANS_BBVA, "silhouette")
        self.view.runs(runs, "silhouette")

    def ej4(self):
        # Ej4 (opcional) - Herramienta IA usada: Claude
        self.view.titulo("Ej4")
        hecho = leer_gold(self.spark, config.GOLD_HECHO)
        bbva = datos_empresa(hecho, config.TICKER_BBVA, config.FEATURES_EJ1)
        # Se guardan las features de los ejercicios anteriores (clave Ticker, Fecha + Open, Close)
        # como versión 1 del feature store y se recuperan desde ahí.
        guardar_version(self.spark, bbva, config.FEATURE_STORE, config.VERSION_FEATURES_EJ4)
        features_v1 = leer_version(self.spark, config.FEATURE_STORE, config.VERSION_FEATURES_EJ4)
        # Se entrena con los datos originales y se reentrena con la versión recuperada (mismo k y seed).
        _, pred_original = entrenar_kmeans(bbva, config.FEATURES_EJ1, config.K_EJ4, config.SEED)
        _, pred_v1 = entrenar_kmeans(features_v1, config.FEATURES_EJ1, config.K_EJ4, config.SEED)
        self.view.valor("Silhouette con los datos originales", silhouette(pred_original))
        self.view.valor("Silhouette con las features versión 1", silhouette(pred_v1))
        self.view.valor("Días con distinto cluster", dias_con_distinto_cluster(pred_original, pred_v1))
        # Por qué favorece la reproducibilidad: el modelo depende de los datos con los que se
        # entrena, del código y del azar. La versión 1 congela los datos exactos (qué días y qué
        # columnas) aunque la gold cambie o se recalcule mañana; la seed fija elimina el azar de
        # KMeans. Con las dos cosas, reentrenar sobre la versión 1 da la misma silhouette y cada
        # día cae en el mismo cluster (0 diferencias), así un modelo se puede recrear y auditar.

    def ej5(self):
        # Ej5 (opcional) - Herramienta IA usada: Claude
        self.view.titulo("Ej5")
        # Misma búsqueda que en el Ej2 (KMeans + Hyperopt + silhouette) pero con otras columnas de
        # BBVA: variación diaria y rango intradía (%), que describen cómo se movió cada día en vez
        # de su nivel de precio. Se guarda en un experimento distinto.
        hecho = leer_gold(self.spark, config.GOLD_HECHO)
        bbva_variacion = datos_empresa(hecho, config.TICKER_BBVA, config.FEATURES_EJ5)
        usar_experimento(config.EXPERIMENTO_EJ5)
        buscar_k_hyperopt(bbva_variacion, config.FEATURES_EJ5, config.K_MIN_EJ2, config.K_MAX_EJ2,
                          config.MAX_EVALS_EJ2, config.SEED)
        # Lista de experimentos y mejor run de cada uno (mayor silhouette).
        self.view.mejores_runs(mejores_runs("silhouette"), "silhouette")
        self.view.mensaje(
            "Comentario: cada experimento agrupa las runs de un mismo problema (mismo algoritmo y "
            "mismos datos), así sus métricas son comparables entre sí y el mejor k de cada uno se "
            "elige sin mezclarse con los demás. Al separar los datos en experimentos distintos se "
            "puede comparar el mismo KMeans sobre Open/Close y sobre variación/rango: cada uno tiene "
            "su propio historial y su mejor run, y se ve que con otros datos cambian el k elegido y "
            "la calidad de la agrupación.")


#========================================================================================
#===================CLASE 2 - Aprendizaje no supervisado (DBSCAN, GMM) ==================
#========================================================================================

    def ej6(self):
        # Ej6 - Herramienta IA usada: Claude
        self.view.titulo("Ej6")
        # Todas las filas empresa-día de la gold (35 empresas) con Return, RangeLow y RangeOpen.
        hecho = variables_derivadas(leer_gold(self.spark, config.GOLD_HECHO))
        matriz = matriz_correlacion(hecho, config.COLUMNAS_CORRELACION_EJ6)
        self.view.matriz(config.COLUMNAS_CORRELACION_EJ6, matriz)
        # Comentario y justificación de las variables elegidas: en el informe (pdf).

    def ej7(self):
        # Ej7 - Herramienta IA usada: Claude
        self.view.titulo("Ej7")
        # Correlación a nivel diario: cada fila empresa-día tal cual, sin agrupar.
        hecho = variables_derivadas(leer_gold(self.spark, config.GOLD_HECHO))
        r = correlacion(hecho, "Sentimiento", "Return")
        self.view.valor("Correlación diaria entre Sentimiento y Return", r)
        # Comentario sobre la relación entre sentimiento del sector y retorno del día: en el informe (pdf).


#========================================================================================
#===================CLASE 3 - Análisis y Power BI =======================================
#========================================================================================
