"""Reproducibilidad: fija la semilla de las librerías de Python que usan azar.

Los algoritmos de Spark MLlib (KMeans, GaussianMixture...) NO usan estas semillas:
tienen su propio parámetro seed, que se fija al crearlos (ver src/modelo/ml/pipelines.py).
"""
import random

import numpy as np


# Usado en: Ej1, Ej2
def fijar_semillas(seed):
    """Fija la semilla del módulo random de Python y de numpy."""
    random.seed(seed)
    np.random.seed(seed)
