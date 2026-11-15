# Práctica 3 – Aprendizaje no supervisado y MLOps con Spark

Sistema MLOps sobre los datos del IBEX-35 (capa gold de la Práctica 2): pipeline de Spark MLlib,
feature store versionado, búsqueda de hiperparámetros con Hyperopt y registro de experimentos con MLflow.

## Estructura (patrón MVC)

    main.py                        punto de entrada
    data/gold/                     capa gold de la Práctica 2 (Parquet, solo lectura)
        hecho_cotizacion/  dim_empresa/  dim_fecha/
    data/lake/features/            feature store (Delta + columna version), se genera al ejecutar
    mlflow.db, mlruns/             historial de MLflow (runs y modelos), se genera al ejecutar
    src/
        controlador/ml_controller.py   un método por ejercicio, separados por clase
        vista/ml_view.py               solo imprime
        modelo/
            comun/     config.py, spark_session.py, semillas.py
            ml/        pipelines.py, evaluacion.py, tuning.py
            datos.py   feature_store.py   tracking.py

## Ejecución

    pip install -r requirements.txt
    python main.py            # todos los ejercicios
    python main.py ej1 ej2    # solo los indicados

Para ver los experimentos en la interfaz de MLflow (desde la raíz del proyecto):

    mlflow ui
