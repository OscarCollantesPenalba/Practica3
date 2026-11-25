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

    def matriz(self, columnas, matriz):
        """Imprime una matriz cuadrada con el nombre de las columnas en filas y columnas."""
        ancho = max(len(c) for c in columnas) + 2
        print(" " * ancho + "".join(f"{c:>{ancho}}" for c in columnas))
        for nombre, fila in zip(columnas, matriz):
            print(f"{nombre:<{ancho}}" + "".join(f"{v:>{ancho}.3f}" for v in fila))

    def run(self, row, metrica):
        """Imprime una run: id, parámetros almacenados y valor de la métrica."""
        # Orden alfabético: MLflow no conserva el orden en que se registraron los parámetros.
        columnas_params = sorted(c for c in row.index if c.startswith("params."))
        params = ", ".join(f"{c.replace('params.', '')}: {row[c]}" for c in columnas_params)
        print(f"Run ID: {row['run_id']}, {params}, {metrica}: {row[f'metrics.{metrica}']}")

    def runs(self, runs_df, metrica):
        """Imprime todas las runs de un DataFrame de MLflow."""
        for _, row in runs_df.iterrows():
            self.run(row, metrica)

    def mejores_runs(self, mejores, metrica):
        """Imprime la lista de experimentos y la mejor run de cada uno."""
        self.valor("Experimentos disponibles", ", ".join(nombre for nombre, _ in mejores))
        for nombre, mejor in mejores:
            print(f"Experimento {nombre}:")
            if mejor is None:
                print("  (sin runs)")
            else:
                print("  ", end="")
                self.run(mejor, metrica)
