class MLView:
    """Capa de vista: solo imprime por consola, no calcula ni transforma datos."""

    def titulo(self, ejercicio):
        print(ejercicio)

    def esquema(self, df):
        df.printSchema()

    def filas(self, df, n=5, truncate=True):
        df.show(n, truncate=truncate)

    def valor(self, etiqueta, valor):
        print(f"{etiqueta}: {valor}")

    def mensaje(self, texto):
        print(texto)

    def seccion(self, texto):
        print(f"----- {texto} -----")

    def runs(self, runs_df, metrica):
        """Imprime cada run: id, parámetros almacenados y valor de la métrica."""
        # Orden alfabético: MLflow no conserva el orden en que se registraron los parámetros.
        columnas_params = sorted(c for c in runs_df.columns if c.startswith("params."))
        for _, row in runs_df.iterrows():
            params = ", ".join(f"{c.replace('params.', '')}: {row[c]}" for c in columnas_params)
            print(f"Run ID: {row['run_id']}, {params}, {metrica}: {row[f'metrics.{metrica}']}")
