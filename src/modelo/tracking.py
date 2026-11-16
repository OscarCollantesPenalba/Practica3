"""MLflow: experimentos, registro de runs y consulta de resultados.
No se fija la ruta del tracking (set_tracking_uri): MLflow usa su ubicación por defecto
en la carpeta desde la que se ejecuta el proyecto."""
import mlflow
import mlflow.spark


# Usado en: Ej1, Ej2
def usar_experimento(experimento):
    """Crea el experimento si no existe y lo deja activo para las siguientes runs."""
    mlflow.set_experiment(experimento)


# Usado en: Ej1, Ej2
def registrar_run(parametros, metricas, modelo):
    """Crea una run en el experimento activo con sus parámetros, métricas y el modelo entrenado."""
    with mlflow.start_run() as run:
        for nombre, valor in parametros.items():
            mlflow.log_param(nombre, valor)
        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)
        mlflow.spark.log_model(modelo, "modelo")
    return run.info.run_id


# Usado en: Ej3
def runs_ordenadas(experimento, metrica, ascendente=False):
    """Recupera todas las runs de un experimento ordenadas por una métrica.
    Devuelve un DataFrame de pandas con columnas run_id, params.<nombre>, metrics.<nombre>...
    ascendente=False: de mayor a menor (silhouette); True: de menor a mayor (SSE)."""
    experiment = mlflow.get_experiment_by_name(experimento)
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return runs_df.sort_values(by=f"metrics.{metrica}", ascending=ascendente)
