# main.py - punto de entrada
# Uso:
#   python main.py            -> ejecuta todos los ejercicios implementados
#   python main.py ej1 ej2    -> ejecuta solo los ejercicios indicados
import sys

from src.controlador.ml_controller import MLController

if __name__ == "__main__":
    MLController().run(sys.argv[1:] or None)
