"""MLflow: experimentos, registro de runs y consulta de resultados.
No se fija la ruta del tracking (set_tracking_uri): MLflow usa su ubicación por defecto
en la carpeta desde la que se ejecuta el proyecto."""
import mlflow
import mlflow.spark


# Usado en: Ej1
def usar_experimento(experimento):
    """Crea el experimento si no existe y lo deja activo para las siguientes runs."""
    mlflow.set_experiment(experimento)


# Usado en: Ej1
def registrar_run(parametros, metricas, modelo):
    """Crea una run en el experimento activo con sus parámetros, métricas y el modelo entrenado."""
    with mlflow.start_run() as run:
        for nombre, valor in parametros.items():
            mlflow.log_param(nombre, valor)
        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)
        mlflow.spark.log_model(modelo, "modelo")
    return run.info.run_id
